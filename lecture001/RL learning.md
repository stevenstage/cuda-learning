# Cuda Mode Lecture 1
## How to profile CUDA kernels in PyTorch
Here gives my self-learning notes on CUDA and Triton. As we all know, coding needs to give feedback directly, so I make my determination to run the code and write some feelings and experiences.
### Simple example
Here is a matrix square example.
```python
def time_pytorch_function(func, input):
    # CUDA IS ASYNC so can't use python time module
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    # Warmup
    for _ in range(5):
        func(input)

    start.record()
    func(input)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end)
```
Paying attention to the difference between cuda time and pytorch time, if you'd like to test cuda time, it should be note to redefine

then turn to an operation with torch square, and look the visualization:

<p align="center">
  <img src="https://github.com/stevenstage/cuda-learning/blob/main/image/lecture_001/2.png" width="800px"/>
</p>


so the torch.profiler.profile() is essential to visualize. It clearly shows the time required for some operations. It can be found that aten::pow takes the longest CPU time, and aten::squre takes the longest GPU time, which stands for power and square, respectively

### torch profiler
Next, we will use profiler code to observe GPU operations in a more fine-grained manner
```python
with torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ],

    schedule=torch.profiler.schedule(
        wait=1,
        warmup=1,
        active=2,
        repeat=1),
```
We used the ProfilerActivity to further observe the time and memory usage of each operation, as shown in the figure below. Overall, the first two sections help you visualize the process by getting familiar with the profiler
<p align="center">
  <img src="https://github.com/stevenstage/cuda-learning/blob/main/image/lecture_001/1.png" width="800px"/>
</p>

### Custom cpp extensions
This part is also relatively simple. Since cuda programming requires c++ and its corresponding compilation environment, there is already a load_inline library in python. You can compile c++ code according to the following code format. In particular, I removed build_directory='./tmp' from the source code, which means downloading the cpp environment based on your own environment.
```python
import torch
from torch.utils.cpp_extension import load_inline

cpp_source = """
std::string hello_world() {
  return "Hello World!";
}
"""

my_module = load_inline(
    name='my_module',
    cpp_sources=[cpp_source],
    functions=['hello_world'],
    verbose=True
)

print(my_module.hello_world())
```
<p align="center">
  <img src="https://github.com/stevenstage/cuda-learning/blob/main/image/lecture_001/3.png" width="800px"/>
</p>

### Integrate a triton kernel
Due to triton's strong ease of use and operability, I will emphasize this part in particular. Triton is a "CUDA alternative for deep learning researchers", focusing on the high-performance implementation of individual operators on gpus. In Python, the @triton.jit decorator compiles on the spot and caches PTX after the first call. And through torch.empty_strided(...)" The obtained CUdeviceptr is directly fed to the Triton kernel.

Triton can accelerate the traditional path of "PyTorch eager → cuBLAS → kernel launch" because it has brought "operator fusion + tile/ block-level automatic tuning + lightweight compilation" down to the JIT stage. This thus eliminates a significant amount of overhead related to Python interpreters, framework scheduling, global synchronization, and memory round trips

But unfortunately, when I tried to use the triton code in the course myself, I got the same result as slide: it didn't make a difference from the traditional torch. Similarly, the problem occurred in the parameter "BLOCK WISE". So I modified the source code and added the function of automatic integration of BLOCK WISE
```python
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=16),
        triton.Config({'BLOCK_SIZE': 8192}, num_warps=16),
        # You can also try different num_warps for the same BLOCK_SIZE
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=16),
    ],
    key=['n_cols'],  # Auto-tune based on number of columns
)
```

The other code is basically modeled after the original code, and the performance has been significantly improved:
<p align="center">
  <img src="https://github.com/stevenstage/cuda-learning/blob/main/image/lecture_001/square()performance.png" width="800px"/>
</p>

### ncu profiler
The last part, ncu, is very important, but it requires you to have your own GPU to operate. Therefore, I won't show your achievements separately. Just look at the course and slides


