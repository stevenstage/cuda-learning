import time
import torch
import triton
import triton.language as tl
from PIL import Image
import torchvision.transforms as T
import os
import fire

@triton.jit
def rgb2gray_kernel_optimized_fixed(
    out_ptr,    # uint8* (H*W)
    in_ptr,     # uint8* (H*W*3)
    H,
    W,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    total = H * W
    mask = offset < total
    
    # Load RGB values separately (avoiding non-power-of-2 arange)
    rgb_offset = offset * 3
    
    # Load R, G, B channels separately
    r = tl.load(in_ptr + rgb_offset, mask=mask, other=0.0).to(tl.float32)
    g = tl.load(in_ptr + rgb_offset + 1, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(in_ptr + rgb_offset + 2, mask=mask, other=0.0).to(tl.float32)
    
    # Compute grayscale using standard luminance formula
    gray = 0.299 * r + 0.587 * g + 0.114 * b
    gray = gray.to(tl.uint8)
    
    tl.store(out_ptr + offset, gray, mask=mask)

@triton.jit 
def rgb2gray_kernel_coalesced(
    out_ptr,    # uint8* (H*W)
    in_ptr,     # uint8* (H*W*3)  
    H,
    W,
    BLOCK_SIZE: tl.constexpr
):
    """Version with better memory coalescing"""
    pid = tl.program_id(0)
    
    # Process multiple pixels per thread for better memory efficiency
    base_offset = pid * BLOCK_SIZE
    
    # Load multiple pixels at once
    offsets = base_offset + tl.arange(0, BLOCK_SIZE)
    total_pixels = H * W
    mask = offsets < total_pixels
    
    # Coalesced RGB loading - load 3 consecutive values
    rgb_offsets = offsets * 3
    
    # Load RGB data in chunks for better memory bandwidth utilization
    r_vals = tl.load(in_ptr + rgb_offsets, mask=mask, other=0).to(tl.float32)
    g_vals = tl.load(in_ptr + rgb_offsets + 1, mask=mask, other=0).to(tl.float32)  
    b_vals = tl.load(in_ptr + rgb_offsets + 2, mask=mask, other=0).to(tl.float32)
    
    # Optimized grayscale computation with bit shifts for faster multiplication
    # 0.299 ≈ 77/256, 0.587 ≈ 150/256, 0.114 ≈ 29/256
    gray = ((77 * r_vals + 150 * g_vals + 29 * b_vals) + 128).to(tl.int32) >> 8
    gray = gray.to(tl.uint8)
    
    tl.store(out_ptr + offsets, gray, mask=mask)

@triton.jit
def rgb2gray_kernel_fast(
    out_ptr,
    in_ptr, 
    H,
    W,
    BLOCK_SIZE: tl.constexpr
):
    """Fastest version with manual loop unrolling and optimizations"""
    pid = tl.program_id(0)
    
    # Calculate pixel indices for this block
    pixel_start = pid * BLOCK_SIZE
    pixel_indices = pixel_start + tl.arange(0, BLOCK_SIZE)
    total_pixels = H * W
    mask = pixel_indices < total_pixels
    
    # Calculate RGB memory offsets
    rgb_indices = pixel_indices * 3
    
    # Load RGB channels separately for better cache utilization
    r = tl.load(in_ptr + rgb_indices, mask=mask, other=0.0).to(tl.float32)
    g = tl.load(in_ptr + rgb_indices + 1, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(in_ptr + rgb_indices + 2, mask=mask, other=0.0).to(tl.float32)
    
    # Fast grayscale conversion using integer approximation
    # Multiply by 256 to avoid floating point, then shift
    gray = ((77 * r + 150 * g + 29 * b) + 128).to(tl.int32) >> 8  # Approximates 0.299, 0.587, 0.114
    
    # Clamp to uint8 range and convert
    gray = tl.where(gray > 255, 255, gray)
    gray = tl.where(gray < 0, 0, gray).to(tl.uint8)
    
    tl.store(out_ptr + pixel_indices, gray, mask=mask)

def rgb2gray_triton_optimized(img_path: str, out_path: str = None, version: str = "fast"):
    """
    Optimized Triton RGB to grayscale conversion
    
    Args:
        img_path: Input image path
        out_path: Output path (optional)
        version: Kernel version to use ('optimized', 'coalesced', 'fast')
    """
    cpu_start = time.perf_counter()
    
    # Load and prepare image
    img = Image.open(img_path).convert("RGB")
    tensor = T.ToTensor()(img) * 255
    tensor = tensor.byte().permute(1, 2, 0).cuda()
    H, W, _ = tensor.shape
    
    # Flatten input for processing
    flat_in = tensor.reshape(-1).contiguous()  # Ensure contiguous memory
    flat_out = torch.empty(H * W, dtype=torch.uint8, device="cuda")
    
    # Choose kernel version
    kernel_map = {
        "optimized": rgb2gray_kernel_optimized_fixed,
        "coalesced": rgb2gray_kernel_coalesced, 
        "fast": rgb2gray_kernel_fast
    }
    kernel = kernel_map.get(version, rgb2gray_kernel_fast)
    
    # Optimize block size based on image size
    total_pixels = H * W
    if total_pixels < 100000:      # Small images
        BLOCK_SIZE = 256
    elif total_pixels < 1000000:   # Medium images  
        BLOCK_SIZE = 512
    else:                          # Large images
        BLOCK_SIZE = 1024
    
    # Calculate grid size
    grid = lambda meta: (triton.cdiv(total_pixels, meta["BLOCK_SIZE"]),)
    
    # Warm up GPU (important for accurate timing)
    for _ in range(3):
        kernel[grid](flat_out, flat_in, H, W, BLOCK_SIZE=BLOCK_SIZE)
    torch.cuda.synchronize()
    
    # Benchmark kernel
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    torch.cuda.synchronize()
    start.record()
    
    # Run kernel multiple times for stable timing
    num_runs = 10
    for _ in range(num_runs):
        kernel[grid](flat_out, flat_in, H, W, BLOCK_SIZE=BLOCK_SIZE)
    
    end.record()
    torch.cuda.synchronize()
    
    gpu_ms = start.elapsed_time(end) / num_runs  # Average time
    
    # Convert result and save
    gray_img = flat_out.view(H, W).cpu().numpy()
    
    if out_path is None:
        base, ext = os.path.splitext(img_path)
        out_path = f"{base}_gray_{version}{ext}"
    
    Image.fromarray(gray_img, mode="L").save(out_path)
    
    cpu_ms = (time.perf_counter() - cpu_start) * 1000
    
    print(f"Version: {version}")
    print(f"Image size: {W}x{H} ({total_pixels:,} pixels)")
    print(f"Block size: {BLOCK_SIZE}")
    print(f"GPU kernel time: {gpu_ms:.3f} ms")
    print(f"Total CPU time: {cpu_ms:.3f} ms") 
    print(f"Throughput: {total_pixels / (gpu_ms / 1000) / 1e6:.1f} Mpixels/s")
    print(f"Saved to: {out_path}")

def benchmark_all_versions(img_path: str):
    """Benchmark all kernel versions"""
    print("Benchmarking all kernel versions...\n")
    
    versions = ["optimized", "coalesced", "fast"]
    for version in versions:
        print(f"=== {version.upper()} VERSION ===")
        rgb2gray_triton_optimized(img_path, version=version)
        print()

if __name__ == "__main__":
    fire.Fire(rgb2gray_triton_optimized)
