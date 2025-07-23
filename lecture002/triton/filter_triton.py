import torch
import triton
import time
import triton.language as tl
from PIL import Image
import fire
import torchvision.transforms as T
import os

# Compile kernels once at module load time
_compiled_kernels = {}

@triton.jit
def mean_filter_kernel_fast(
    out_ptr, in_ptr,
    C, H, W,
    radius,
    BLOCK_SIZE: tl.constexpr,
    RADIUS: tl.constexpr  # Compile-time constant for better optimization
):
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    total = C * H * W
    mask = offset < total
    
    # Calculate coordinates - more efficient division
    c = offset // (H * W)
    rem = offset % (H * W)
    h = rem // W
    w = rem % W
    
    # Pre-calculate once
    kernel_size = (2 * RADIUS + 1) * (2 * RADIUS + 1)
    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    
    # Unroll for common small radii - this is much faster
    if RADIUS == 1:
        # Manually unrolled 3x3 - no loops at all
        # Process each of the 9 neighbors explicitly
        for dh in tl.static_range(-1, 2):
            for dw in tl.static_range(-1, 2):
                nh = h + dh
                nw = w + dw
                inside = (nh >= 0) & (nh < H) & (nw >= 0) & (nw < W)
                idx = c * H * W + nh * W + nw
                val = tl.load(in_ptr + idx, mask=mask & inside, other=0.0).to(tl.float32)
                acc = tl.where(inside & mask, acc + val, acc)
    
    elif RADIUS == 2:
        # 5x5 unrolled
        for dh in tl.static_range(-2, 3):
            for dw in tl.static_range(-2, 3):
                nh = h + dh
                nw = w + dw
                inside = (nh >= 0) & (nh < H) & (nw >= 0) & (nw < W)
                idx = c * H * W + nh * W + nw
                val = tl.load(in_ptr + idx, mask=mask & inside, other=0.0).to(tl.float32)
                acc = tl.where(inside & mask, acc + val, acc)
    
    elif RADIUS == 3:
        # 7x7 unrolled
        for dh in tl.static_range(-3, 4):
            for dw in tl.static_range(-3, 4):
                nh = h + dh
                nw = w + dw
                inside = (nh >= 0) & (nh < H) & (nw >= 0) & (nw < W)
                idx = c * H * W + nh * W + nw
                val = tl.load(in_ptr + idx, mask=mask & inside, other=0.0).to(tl.float32)
                acc = tl.where(inside & mask, acc + val, acc)
    
    else:
        # General case - only for large radii
        for dh in range(-RADIUS, RADIUS + 1):
            for dw in range(-RADIUS, RADIUS + 1):
                nh = h + dh
                nw = w + dw
                inside = (nh >= 0) & (nh < H) & (nw >= 0) & (nw < W)
                idx = c * H * W + nh * W + nw
                val = tl.load(in_ptr + idx, mask=mask & inside, other=0.0).to(tl.float32)
                acc = tl.where(inside & mask, acc + val, acc)
    
    # Simplified boundary handling - good enough for most cases
    out_val = (acc / kernel_size).to(tl.uint8)
    tl.store(out_ptr + offset, out_val, mask=mask)

@triton.jit
def mean_filter_2d_fast(
    out_ptr, in_ptr,
    H, W, radius,
    BLOCK_H: tl.constexpr,
    BLOCK_W: tl.constexpr
):
    """Optimized 2D version with better memory access patterns"""
    pid_h = tl.program_id(0)
    pid_w = tl.program_id(1)
    
    h_start = pid_h * BLOCK_H
    w_start = pid_w * BLOCK_W
    
    h_range = tl.arange(0, BLOCK_H)
    w_range = tl.arange(0, BLOCK_W)
    
    h_coords = h_start + h_range
    w_coords = w_start + w_range
    
    h_mask = h_coords < H
    w_mask = w_coords < W
    
    # Vectorized accumulation
    acc = tl.zeros([BLOCK_H, BLOCK_W], dtype=tl.float32)
    kernel_size = (2 * radius + 1) * (2 * radius + 1)
    
    # More efficient nested loop with better memory patterns
    for dh in range(-radius, radius + 1):
        h_sample = h_coords[:, None] + dh
        h_valid = (h_sample >= 0) & (h_sample < H)
        
        for dw in range(-radius, radius + 1):
            w_sample = w_coords[None, :] + dw
            w_valid = (w_sample >= 0) & (w_sample < W)
            
            valid_mask = h_mask[:, None] & w_mask[None, :] & h_valid & w_valid
            
            # Coalesced memory access
            indices = h_sample * W + w_sample
            values = tl.load(in_ptr + indices, mask=valid_mask, other=0.0).to(tl.float32)
            acc += values
    
    result = (acc / kernel_size).to(tl.uint8)
    
    # Store with proper masking
    output_mask = h_mask[:, None] & w_mask[None, :]
    output_indices = h_coords[:, None] * W + w_coords[None, :]
    tl.store(out_ptr + output_indices, result, mask=output_mask)

def get_compiled_kernel(radius: int):
    """Get pre-compiled kernel for given radius to avoid compilation overhead"""
    if radius not in _compiled_kernels:
        # Pre-compile for common radii
        if radius <= 3:
            _compiled_kernels[radius] = mean_filter_kernel_fast
        else:
            _compiled_kernels[radius] = mean_filter_2d_fast
    return _compiled_kernels[radius]

def mean_filter_triton_cpu_optimized(image: torch.Tensor, radius: int) -> torch.Tensor:
    """
    CPU-optimized version that minimizes Python overhead
    """
    assert image.is_cuda and image.dtype == torch.uint8
    assert radius > 0
    
    C, H, W = image.shape
    
    # Pre-allocate output to avoid allocation overhead
    output = torch.empty_like(image)
    
    # Choose optimal strategy based on problem size
    total_pixels = C * H * W
    
    if radius <= 3 and total_pixels < 4 * 1024 * 1024:  # Small kernels, small-medium images
        # Use optimized 1D version with compile-time radius
        flat_in = image.view(-1)
        flat_out = output.view(-1)
        
        # Optimal block size for small kernels
        BLOCK_SIZE = min(1024, triton.next_power_of_2(total_pixels // 128))
        BLOCK_SIZE = max(BLOCK_SIZE, 128)  # Minimum for occupancy
        
        grid_size = triton.cdiv(total_pixels, BLOCK_SIZE)
        
        # Compile-time radius specialization
        if radius == 1:
            mean_filter_kernel_fast[(grid_size,)](
                flat_out, flat_in, C, H, W, radius, 
                BLOCK_SIZE=BLOCK_SIZE, RADIUS=1
            )
        elif radius == 2:
            mean_filter_kernel_fast[(grid_size,)](
                flat_out, flat_in, C, H, W, radius,
                BLOCK_SIZE=BLOCK_SIZE, RADIUS=2
            )
        elif radius == 3:
            mean_filter_kernel_fast[(grid_size,)](
                flat_out, flat_in, C, H, W, radius,
                BLOCK_SIZE=BLOCK_SIZE, RADIUS=3
            )
            
    else:  # Large kernels or large images
        # Use 2D blocking for better cache locality
        BLOCK_H = BLOCK_W = 16 if radius > 5 else 32
        
        for c in range(C):
            grid = (triton.cdiv(H, BLOCK_H), triton.cdiv(W, BLOCK_W))
            mean_filter_2d_fast[grid](
                output[c], image[c],
                H, W, radius,
                BLOCK_H=BLOCK_H, BLOCK_W=BLOCK_W
            )
    
    return output

# Separable version optimized for CPU overhead
@triton.jit
def separable_horizontal_fast(
    out_ptr, in_ptr, H, W, radius,
    BLOCK_SIZE: tl.constexpr,
    RADIUS: tl.constexpr
):
    pid = tl.program_id(0)
    base_idx = pid * BLOCK_SIZE
    offsets = base_idx + tl.arange(0, BLOCK_SIZE)
    mask = offsets < H * W
    
    h = offsets // W
    w = offsets % W
    
    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    
    # Unrolled for small radii
    if RADIUS <= 3:
        for dw in tl.static_range(-RADIUS, RADIUS + 1):
            nw = w + dw
            valid = (nw >= 0) & (nw < W) & mask
            idx = h * W + nw
            val = tl.load(in_ptr + idx, mask=valid, other=0.0).to(tl.float32)
            acc += tl.where(valid, val, 0.0)
        divisor = 2 * RADIUS + 1
    else:
        # Dynamic version for large radii
        count = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
        for dw in range(-radius, radius + 1):
            nw = w + dw
            valid = (nw >= 0) & (nw < W) & mask
            idx = h * W + nw
            val = tl.load(in_ptr + idx, mask=valid, other=0.0).to(tl.float32)
            acc = tl.where(valid, acc + val, acc)
            count = tl.where(valid, count + 1.0, count)
        divisor = count
    
    result = (acc / divisor).to(tl.uint8)
    tl.store(out_ptr + offsets, result, mask=mask)

@triton.jit
def separable_vertical_fast(
    out_ptr, in_ptr, H, W, radius,
    BLOCK_SIZE: tl.constexpr,
    RADIUS: tl.constexpr
):
    pid = tl.program_id(0)
    base_idx = pid * BLOCK_SIZE
    offsets = base_idx + tl.arange(0, BLOCK_SIZE)
    mask = offsets < H * W
    
    h = offsets // W
    w = offsets % W
    
    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    
    if RADIUS <= 3:
        for dh in tl.static_range(-RADIUS, RADIUS + 1):
            nh = h + dh
            valid = (nh >= 0) & (nh < H) & mask
            idx = nh * W + w
            val = tl.load(in_ptr + idx, mask=valid, other=0.0).to(tl.float32)
            acc += tl.where(valid, val, 0.0)
        divisor = 2 * RADIUS + 1
    else:
        count = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
        for dh in range(-radius, radius + 1):
            nh = h + dh
            valid = (nh >= 0) & (nh < H) & mask
            idx = nh * W + w
            val = tl.load(in_ptr + idx, mask=valid, other=0.0).to(tl.float32)
            acc = tl.where(valid, acc + val, acc)
            count = tl.where(valid, count + 1.0, count)
        divisor = count
    
    result = (acc / divisor).to(tl.uint8)
    tl.store(out_ptr + offsets, result, mask=mask)

def mean_filter_separable_fast(image: torch.Tensor, radius: int) -> torch.Tensor:
    """CPU-optimized separable filter"""
    assert image.is_cuda and image.dtype == torch.uint8
    
    C, H, W = image.shape
    temp = torch.empty_like(image)
    output = torch.empty_like(image)
    
    BLOCK_SIZE = 1024
    grid_size = triton.cdiv(H * W, BLOCK_SIZE)
    
    # Process all channels with minimal Python overhead
    for c in range(C):
        if radius <= 3:
            # Compile-time specialization
            if radius == 1:
                separable_horizontal_fast[(grid_size,)](temp[c], image[c], H, W, radius, BLOCK_SIZE, RADIUS=1)
                separable_vertical_fast[(grid_size,)](output[c], temp[c], H, W, radius, BLOCK_SIZE, RADIUS=1)
            elif radius == 2:
                separable_horizontal_fast[(grid_size,)](temp[c], image[c], H, W, radius, BLOCK_SIZE, RADIUS=2)
                separable_vertical_fast[(grid_size,)](output[c], temp[c], H, W, radius, BLOCK_SIZE, RADIUS=2)
            elif radius == 3:
                separable_horizontal_fast[(grid_size,)](temp[c], image[c], H, W, radius, BLOCK_SIZE, RADIUS=3)
                separable_vertical_fast[(grid_size,)](output[c], temp[c], H, W, radius, BLOCK_SIZE, RADIUS=3)
        else:
            # General case
            separable_horizontal_fast[(grid_size,)](temp[c], image[c], H, W, radius, BLOCK_SIZE, RADIUS=radius)
            separable_vertical_fast[(grid_size,)](output[c], temp[c], H, W, radius, BLOCK_SIZE, RADIUS=radius)
    
    return output

# Batch processing to amortize overhead
def mean_filter_batch_optimized(images: list, radius: int, method: str = "auto"):
    """Process multiple images in batch to amortize compilation overhead"""
    if not images:
        return []
    
    # Pre-compile kernels
    if method == "auto":
        sample_img = images[0]
        H, W = sample_img.shape[1], sample_img.shape[2]
        method = "separable" if radius > 5 else "optimized"
    
    # Warmup compilation
    warmup_img = images[0]
    if method == "separable":
        _ = mean_filter_separable_fast(warmup_img, radius)
    else:
        _ = mean_filter_triton_cpu_optimized(warmup_img, radius)
    
    # Process batch
    results = []
    for img in images:
        if method == "separable":
            result = mean_filter_separable_fast(img, radius)
        else:
            result = mean_filter_triton_cpu_optimized(img, radius)
        results.append(result)
    
    return results

if __name__ == "__main__":
    def cli(img_path: str, radius: int = 3, out_path: str = None, method: str = "auto", 
            batch_size: int = 1):
        """
        method: "optimized", "separable", "auto"
        batch_size: Process multiple copies to show batching benefits
        """
        
        img = Image.open(img_path).convert("RGB")
        tensor = T.ToTensor()(img) * 255
        tensor = tensor.byte().cuda()
        H, W = tensor.shape[1], tensor.shape[2]
        
        # Auto-select method
        if method == "auto":
            if radius > 5 or (radius > 3 and H * W > 1024 * 1024):
                method = "separable"
            else:
                method = "optimized"
            print(f"Auto-selected: {method} (radius={radius}, size={H}x{W})")
        
        # Prepare batch if requested
        batch = [tensor.clone() for _ in range(batch_size)]
        
        # Method selection
        if method == "separable":
            filter_func = lambda x: mean_filter_separable_fast(x, radius)
        else:
            filter_func = lambda x: mean_filter_triton_cpu_optimized(x, radius)
        
        # Warmup - crucial for Triton
        print("Warming up...")
        for _ in range(5):  # More warmup iterations
            _ = filter_func(tensor)
        torch.cuda.synchronize()
        
        # Benchmark
        torch.cuda.empty_cache()  # Clear cache
        
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        cpu_start = time.perf_counter()
        start_event.record()
        
        if batch_size == 1:
            result = filter_func(tensor)
        else:
            results = mean_filter_batch_optimized(batch, radius, method)
            result = results[0]  # Use first for saving
        
        end_event.record()
        torch.cuda.synchronize()
        cpu_end = time.perf_counter()
        
        gpu_ms = start_event.elapsed_time(end_event)
        cpu_wall_ms = (cpu_end - cpu_start) * 1000
        
        # Save result
        if out_path is None:
            base, ext = os.path.splitext(img_path)
            out_path = f"{base}_blur_fast_{method}_r{radius}{ext}"
        
        result_np = result.permute(1, 2, 0).cpu().numpy()
        Image.fromarray(result_np).save(out_path)
        
        # Performance analysis
        total_pixels = tensor.numel() * batch_size
        
        print(f"\n=== Optimized Performance Results ===")
        print(f"Batch size: {batch_size}")
        print(f"Image size: {H}x{W} ({tensor.numel():,} pixels per image)")
        print(f"Filter radius: {radius}")
        print(f"Method: {method}")
        print(f"GPU time: {gpu_ms:.3f} ms")
        print(f"CPU wall time: {cpu_wall_ms:.3f} ms") 
        print(f"Overhead: {cpu_wall_ms - gpu_ms:.3f} ms ({(cpu_wall_ms/gpu_ms - 1)*100:.1f}%)")
        
        # Optimization suggestions
        overhead_ratio = cpu_wall_ms / gpu_ms
        if overhead_ratio > 2.5:
            print(f"\n⚠️  High CPU overhead ({overhead_ratio:.1f}x)!")
            print("💡 Suggestions:")
            print("   - Use batch processing for multiple images")
            print("   - Consider pre-warming kernels in production")
            if method == "separable" and radius <= 5:
                print("   - Try 'optimized' method for smaller radii")
        elif overhead_ratio < 1.5:
            print(f"\n✅ Good CPU efficiency ({overhead_ratio:.1f}x overhead)")
        
    fire.Fire(cli)
