import torch
import triton
import triton.language as tl
from PIL import Image
import torchvision.transforms as T
import os
import fire   # pip install fire

@triton.jit
def _rgb2gray_kernel(
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
    offset = tl.where(mask, offset, 0)

    # 计算 RGB 偏移
    r_off = offset * 3
    g_off = r_off + 1
    b_off = r_off + 2

    r = tl.load(in_ptr + r_off, mask=mask).to(tl.float32)
    g = tl.load(in_ptr + g_off, mask=mask).to(tl.float32)
    b = tl.load(in_ptr + b_off, mask=mask).to(tl.float32)

    gray = 0.21 * r + 0.71 * g + 0.07 * b
    gray = gray.to(tl.uint8)
    tl.store(out_ptr + offset, gray, mask=mask)

def rgb2gray_triton(img_path: str, out_path: str = None):
    # 1) read image -> [H,W,3] uint8 CUDA
    img = Image.open(img_path).convert("RGB")
    tensor = T.ToTensor()(img)              # [3,H,W] float32 [0,1]
    tensor = (tensor * 255).byte()          # [3,H,W] uint8
    tensor = tensor.permute(1, 2, 0).cuda() # [H,W,3] CUDA uint8

    H, W, _ = tensor.shape
    flat_in = tensor.reshape(-1)               # [H*W*3]
    flat_out = torch.empty(H * W, dtype=torch.uint8, device="cuda")

    # 2) start Triton kernel
    BLOCK_SIZE = 256
    num_pixels = H * W
    grid = lambda meta: (triton.cdiv(num_pixels, meta["BLOCK_SIZE"]),)

    _rgb2gray_kernel[grid](
        flat_out, flat_in,
        H, W,                       # directly H, W
        BLOCK_SIZE=BLOCK_SIZE
    )

    # 3) reload
    gray_img = flat_out.view(H, W).cpu().numpy()
    if out_path is None:
        base, ext = os.path.splitext(img_path)
        out_path = f"{base}_gray{ext}"
    Image.fromarray(gray_img, mode="L").save(out_path)
    print(f"saved grayscale image to {out_path}")


if __name__ == "__main__":
    fire.Fire(rgb2gray_triton)
