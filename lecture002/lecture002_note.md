# Cuda Mode Lecture 2
## PPMP book chapter 1-3
The slides already give you all the info you need about the principles, so this tutorial will focus on how to do the experiments and record the process to help get you started quickly.

### Ex1: vector addition
The implementation of this function is found in the [vector_add.cu](https://github.com/stevenstage/cuda-learning/blob/main/lecture002/vector_add.cu) file. It should be noted that the CUDA kernel is written directly in C++, and consequently, during the compilation process, the initial step involves generating a.out, which then requires further compilation to yield the desired result.

The following is the compilation statement for the cu file:
```
  nvcc vector_add.cu
```
Before proceeding, please ensure that PyTorch and the CUDA toolkit (nvcc compiler) are installed.

After generating the a.out file, you can directly compile the a.out file using the following statement:
```
  ./a.out
```
The generated results are shown below：
<p align="center">
  <img src="https://github.com/stevenstage/cuda-learning/blob/main/image/lecture_002/1.png" width="800px"/>
</p>

In order to comprehend the underlying causes of this outcome, a meticulous and methodical examination of the code is imperative. Initially, the process of this CUDA kernel entails the creation of three arrays, designated A, B, and C, with each array comprising 1,000 elements. Subsequently, a copy of A and B is transferred to the graphics memory. Following this, the kernel function is invoked, resulting in the calculation of C. Finally, the resultant value is printed. The kernel function in question performs 1,000 additions in parallel on the GPU：
```
  C[i] = A[i] + B[i];
```
Proceeding with the detailed examination of the process, the initial phase of the kernel function vecAddKernel is as follows:
```c++
__global__ void vecAddKernel(float *A, float *B, float *C, int n) {
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  if (i < n) {
    C[i] = A[i] + B[i];
  }
}
```
* __global__ tells nvcc that this is a GPU entry point, and the CPU calls it using <<< >>>
* Each thread first calculates its own global ID, then calls the addition CUDA kernel
  
The second part is the host-side function vecAdd, which is mainly used for CPU scheduling. The detailed process is as follows:
```c++
void vecAdd(float *A, float *B, float *C, int n) {
  float *A_d, *B_d, *C_d;
  size_t size = n * sizeof(float);

  cudaMalloc((void **)&A_d, size);
  cudaMalloc((void **)&B_d, size);
  cudaMalloc((void **)&C_d, size);

  cudaMemcpy(A_d, A, size, cudaMemcpyHostToDevice);
  cudaMemcpy(B_d, B, size, cudaMemcpyHostToDevice);

  const unsigned int numThreads = 256;
  unsigned int numBlocks = cdiv(n, numThreads);

  vecAddKernel<<<numBlocks, numThreads>>>(A_d, B_d, C_d, n);
  gpuErrchk(cudaPeekAtLastError());
  gpuErrchk(cudaDeviceSynchronize());

  cudaMemcpy(C, C_d, size, cudaMemcpyDeviceToHost);

  cudaFree(A_d);
  cudaFree(B_d);
  cudaFree(C_d);
}
```
* cudaMalloc is used for memory allocation.
* By defining a constant as a block setting 256 threads, and using **cdiv** to ensure full coverage without any remainder.
* **vecAddKernel** and **gpuErrchk** are used for kernel function launch and error checking, respectively. Error checking and protection are performed throughout the entire process from startup to D2H copy.

The third part is error checking：
```c++
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
```
If an error occurs, it will report GPUassert: 
```
GPUassert: <error string>    filename     line number
```
which is convenient for quick location.

Finally, we call the main function. The core part is actually this section:
```c++
// generate some dummy vectors to add
  for (int i = 0; i < n; i += 1) {
    A[i] = float(i);
    B[i] = A[i] / 1000.0f;
  }

```
It is the following expression:
```c++
C[i] = i * 1.001
```

### Ex2: image become gray
After becoming familiar with the syntax of CUDA kernel functions, we can utilize this custom kernel function to implement related functionality, i.e., combining C++ and Python to maximise the use of GPU memory to create hardware-aligned algorithms.

Hey, so the next experiment we're going to do together is going to be really cool. We're going to convert an image to grayscale. The idea is really simple. All you have to do is convert each pixel's RGB values to grayscale using this formula: (0.21*R + 0.71*G + 0.07*B). This will change the image from a three-channel format to a single-channel format. You'll be pleased to know that the core of the code operation is in [rgb2gray.cu](https://github.com/stevenstage/cuda-learning/blob/main/lecture002/rgb2gray.cu). The Python code operation uses the CUDA kernel function for calling and functionality implementation, as we mentioned in the previous lesson under the name load_inline.

```c++
__global__
void rgb_to_grayscale_kernel(unsigned char* output, unsigned char* input, int width, int height) {
    const int channels = 3;

    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (col < width && row < height) {
        int outputOffset = row * width + col;
        int inputOffset = (row * width + col) * channels;

        unsigned char r = input[inputOffset + 0];   // red
        unsigned char g = input[inputOffset + 1];   // green
        unsigned char b = input[inputOffset + 2];   // blue

        output[outputOffset] = (unsigned char)(0.21f * r + 0.71f * g + 0.07f * b);
    }
}
```
* First, thread index calculation. The code is exactly the same as in the above experiment, except that the input image is two-dimensional data, so both row and column indices need to be written out. 
* Next, memory offset calculation is performed, since the input is three-channel and the output is one-channel.
* Finally, the grayscale conversion formula described above is used to make adjustments.

The next part is the interface with PyTorch. There isn't much technical content here:

```c++
torch::Tensor rgb_to_grayscale(torch::Tensor image) {
    assert(image.device().type() == torch::kCUDA);
    assert(image.dtype() == torch::kByte);

    const auto height = image.size(0);
    const auto width = image.size(1);

    auto result = torch::empty({height, width, 1}, torch::TensorOptions().dtype(torch::kByte).device(image.device()));

    dim3 threads_per_block(16, 16);     // using 256 threads per block
    dim3 number_of_blocks(cdiv(width, threads_per_block.x),
                          cdiv(height, threads_per_block.y));

    rgb_to_grayscale_kernel<<<number_of_blocks, threads_per_block, 0, torch::cuda::getCurrentCUDAStream()>>>(
        result.data_ptr<unsigned char>(),
        image.data_ptr<unsigned char>(),
        width,
        height
    );
```
It is simply a matter of creating the output tensor and then starting the kernel. The output  is shown below:
<p align="center">
  <img src="https://github.com/stevenstage/cuda-learning/blob/main/image/lecture_002/2.png" width="800px"/>
</p>
<p align="center">
  <img src="https://github.com/stevenstage/cuda-learning/blob/main/image/lecture_002/gray_output.png" width="800px"/>  
</p>

### Ex3: Mean Filter
This experiment mainly implements image blurring. Since it is very similar to Ex2, we will only introduce how the kernel function is written here. The implementation of this function is found in the [filter.cu](https://github.com/stevenstage/cuda-learning/blob/main/lecture002/filter.cu) file.
```c++
__global__
void mean_filter_kernel(unsigned char* output, unsigned char* input, int width, int height, int radius) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int channel = threadIdx.z;

    int baseOffset = channel * height * width;
    if (col < width && row < height) {

        int pixVal = 0;
        int pixels = 0;

        for (int blurRow=-radius; blurRow <= radius; blurRow += 1) {
            for (int blurCol=-radius; blurCol <= radius; blurCol += 1) {
                int curRow = row + blurRow;
                int curCol = col + blurCol;
                if (curRow >= 0 && curRow < height && curCol >=0 && curCol < width) {
                    pixVal += input[baseOffset + curRow * width + curCol];
                    pixels += 1;
                }
            }
        }

        output[baseOffset + row * width + col] = (unsigned char)(pixVal / pixels);
    }
}
```
After calculating the thread index and offset, we arrive at the core part of the mean filter: two for functions iterate through a (2*radius+1) × (2*radius+1) square centred on (row, col), accumulate only the pixel values within the boundary, and finally divide by the actual number of accumulated pixels to obtain the average value. Finally, the result is written back directly.

The output is shown below:
<p align="center">
  <img src="https://github.com/stevenstage/cuda-learning/blob/main/image/lecture_002/3.png" width="800px"/>
</p>
<p align="center">
  <img src="https://github.com/stevenstage/cuda-learning/blob/main/image/lecture_002/filter_output.png" width="800px"/>  
</p>
