import torch
import triton
import triton.language as tl
from PIL import Image
import fire   # pip install fire
import torchvision.transforms as T
import os

@triton.jit
def mean_filter_kernel(
    out_ptr, in_ptr,
    C, H, W,
    radius,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    total = C * H * W
    mask = offset < total
    
    c = offset // (H * W)
    rem = offset % (H * W)
    h = rem // W
    w = rem % W
    
    # 向量型累加器
    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    cnt = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    
    for dh in range(-radius, radius + 1):
        for dw in range(-radius, radius + 1):
            nh = h + dh
            nw = w + dw
            inside = (nh >= 0) & (nh < H) & (nw >= 0) & (nw < W)
            idx = c * H * W + nh * W + nw
            val = tl.load(in_ptr + idx, mask=mask & inside, other=0.0).to(tl.float32)
            acc = tl.where(inside, acc + val, acc)
            cnt = tl.where(inside, cnt + 1.0, cnt)
    
    out_val = (acc / cnt).to(tl.uint8)
    tl.store(out_ptr + offset, out_val, mask=mask)

def mean_filter_triton(image: torch.Tensor, radius: int) -> torch.Tensor:
    """
    image: (C, H, W) uint8 CUDA tensor
    radius: int > 0
    return: (C, H, W) uint8 CUDA tensor
    """
    assert image.is_cuda and image.dtype == torch.uint8
    assert radius > 0
    
    C, H, W = image.shape
    flat_in = image.reshape(-1)        # [C*H*W]
    flat_out = torch.empty_like(flat_in)
    
    # 每个像素一个线程，BLOCK_SIZE 可调
    BLOCK_SIZE = 256
    grid = lambda meta: (triton.cdiv(C * H * W, meta['BLOCK_SIZE']),)
    
    mean_filter_kernel[grid](
        flat_out, flat_in,
        C, H, W, radius,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return flat_out.view(C, H, W)

if __name__ == "__main__":
    def cli(img_path: str, radius: int = 3, out_path: str = None):
        img = Image.open(img_path).convert("RGB")
        tensor = T.ToTensor()(img) * 255
        tensor = tensor.byte().cuda()          # [3,H,W]
        
        blurred = mean_filter_triton(tensor, radius)
        
        if out_path is None:
            base, ext = os.path.splitext(img_path)
            out_path = f"{base}_blur_r{radius}{ext}"
        
        # 转回 PIL 保存
        blurred_np = blurred.permute(1, 2, 0).cpu().numpy()  # [H,W,3]
        Image.fromarray(blurred_np).save(out_path)
        print(f"saved blurred image to {out_path}")

    fire.Fire(cli)
