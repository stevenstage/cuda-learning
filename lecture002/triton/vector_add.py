import torch
import triton
import triton.language as tl
import fire # pip install fire  

@triton.jit
def _vecadd_kernel(
    A_ptr, B_ptr, C_ptr,
    n,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < n

    a = tl.load(A_ptr + offset, mask=mask)
    b = tl.load(B_ptr + offset, mask=mask)
    c = a + b
    tl.store(C_ptr + offset, c, mask=mask)

def vecadd_triton(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    assert a.is_cuda and b.is_cuda
    assert a.dtype == b.dtype == torch.float32
    assert a.numel() == b.numel()
    n = a.numel()

    c = torch.empty_like(a)
    BLOCK_SIZE = 256
    grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)

    _vecadd_kernel[grid](
        a, b, c, n,
        BLOCK_SIZE=BLOCK_SIZE
    )
    return c

def main(n: int = 1000):
    a = torch.arange(n, dtype=torch.float32, device="cuda")
    b = a / 1000.0
    c = vecadd_triton(a, b)

    # 打印前 20 个结果验证
    print(c[:20])

if __name__ == "__main__":
    fire.Fire(main)
