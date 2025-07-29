#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// CUDA error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " - " << cudaGetErrorString(error) << std::endl; \
            exit(1); \
        } \
    } while(0)

// Sobel kernels as constant memory
__constant__ int d_sobelX[9] = {-1, 0, 1, -2, 0, 2, -1, 0, 1};
__constant__ int d_sobelY[9] = {-1, -2, -1, 0, 0, 0, 1, 2, 1};

// Image structure
struct Image {
    int width, height, channels;
    std::vector<unsigned char> data;
    
    Image(int w, int h, int c) : width(w), height(h), channels(c) {
        data.resize(w * h * c);
    }
    
    unsigned char getPixel(int x, int y, int c) const {
        if (x < 0 || x >= width || y < 0 || y >= height) return 0;
        return data[(y * width + x) * channels + c];
    }
    
    void setPixel(int x, int y, int c, unsigned char value) {
        if (x >= 0 && x < width && y >= 0 && y < height) {
            data[(y * width + x) * channels + c] = value;
        }
    }
};

// CUDA kernel without shared memory
__global__ void convolutionKernel(unsigned char* input, unsigned char* output, 
                                 int width, int height, int channels) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.z;
    
    if (x >= width || y >= height || c >= channels) return;
    
    float sumX = 0.0f, sumY = 0.0f;
    
    // Apply Sobel convolution
    for (int ky = 0; ky < 3; ky++) {
        for (int kx = 0; kx < 3; kx++) {
            int px = x + kx - 1;
            int py = y + ky - 1;
            
            unsigned char pixel = 0;
            if (px >= 0 && px < width && py >= 0 && py < height) {
                pixel = input[(py * width + px) * channels + c];
            }
            
            int kernelIdx = ky * 3 + kx;
            sumX += pixel * d_sobelX[kernelIdx];
            sumY += pixel * d_sobelY[kernelIdx];
        }
    }
    
    float magnitude = sqrtf(sumX * sumX + sumY * sumY);
    magnitude = fminf(255.0f, magnitude);
    
    output[(y * width + x) * channels + c] = (unsigned char)magnitude;
}

// CUDA kernel with shared memory optimization
__global__ void convolutionKernelShared(unsigned char* input, unsigned char* output, 
                                       int width, int height, int channels) {
    // Shared memory for tile + halo
    __shared__ unsigned char sharedMem[34][34]; // 32x32 + 2-pixel border
    
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.z;
    
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    if (c >= channels) return;
    
    // Load data into shared memory with halo
    // Main tile
    if (x < width && y < height) {
        sharedMem[ty + 1][tx + 1] = input[(y * width + x) * channels + c];
    } else {
        sharedMem[ty + 1][tx + 1] = 0;
    }
    
    // Load halo regions
    // Top halo
    if (ty == 0) {
        int haloY = y - 1;
        if (x < width && haloY >= 0) {
            sharedMem[0][tx + 1] = input[(haloY * width + x) * channels + c];
        } else {
            sharedMem[0][tx + 1] = 0;
        }
    }
    
    // Bottom halo
    if (ty == blockDim.y - 1) {
        int haloY = y + 1;
        if (x < width && haloY < height) {
            sharedMem[ty + 2][tx + 1] = input[(haloY * width + x) * channels + c];
        } else {
            sharedMem[ty + 2][tx + 1] = 0;
        }
    }
    
    // Left halo
    if (tx == 0) {
        int haloX = x - 1;
        if (haloX >= 0 && y < height) {
            sharedMem[ty + 1][0] = input[(y * width + haloX) * channels + c];
        } else {
            sharedMem[ty + 1][0] = 0;
        }
    }
    
    // Right halo
    if (tx == blockDim.x - 1) {
        int haloX = x + 1;
        if (haloX < width && y < height) {
            sharedMem[ty + 1][tx + 2] = input[(y * width + haloX) * channels + c];
        } else {
            sharedMem[ty + 1][tx + 2] = 0;
        }
    }
    
    // Corner halos
    if (tx == 0 && ty == 0) {
        int haloX = x - 1, haloY = y - 1;
        if (haloX >= 0 && haloY >= 0) {
            sharedMem[0][0] = input[(haloY * width + haloX) * channels + c];
        } else {
            sharedMem[0][0] = 0;
        }
    }
    
    __syncthreads();
    
    if (x >= width || y >= height) return;
    
    float sumX = 0.0f, sumY = 0.0f;
    
    // Apply convolution using shared memory
    for (int ky = 0; ky < 3; ky++) {
        for (int kx = 0; kx < 3; kx++) {
            unsigned char pixel = sharedMem[ty + ky][tx + kx];
            int kernelIdx = ky * 3 + kx;
            sumX += pixel * d_sobelX[kernelIdx];
            sumY += pixel * d_sobelY[kernelIdx];
        }
    }
    
    float magnitude = sqrtf(sumX * sumX + sumY * sumY);
    magnitude = fminf(255.0f, magnitude);
    
    output[(y * width + x) * channels + c] = (unsigned char)magnitude;
}

// Load test image
Image loadTestImage() {
    const char* filename = "F1.png";   // ← 换成你的文件名
    int w, h, comp;
    unsigned char* pixels = stbi_load(filename, &w, &h, &comp, 0);
    if (!pixels) {
        std::cerr << "Cannot load " << filename << std::endl;
        exit(1);
    }

    // 若原图是灰度或 RGBA，统一到 3 通道
    int channels = (comp >= 3) ? 3 : comp;
    Image img(w, h, channels);
    std::copy(pixels, pixels + w * h * channels, img.data.begin());

    stbi_image_free(pixels);
    std::cout << "Loaded " << w << "×" << h << " RGB image (" << channels << " ch)" << std::endl;
    return img;
}

// CUDA convolution wrapper
Image cudaConvolution(const Image& input, int blockSize, bool useSharedMemory = false) {
    std::cout << "Starting CUDA convolution with block size " << blockSize << "x" << blockSize;
    if (useSharedMemory) std::cout << " (with shared memory)";
    std::cout << "..." << std::endl;
    
    auto start = std::chrono::high_resolution_clock::now();
    
    Image output(input.width, input.height, input.channels);
    
    // Allocate GPU memory
    unsigned char *d_input, *d_output;
    size_t imageSize = input.width * input.height * input.channels * sizeof(unsigned char);
    
    CUDA_CHECK(cudaMalloc(&d_input, imageSize));
    CUDA_CHECK(cudaMalloc(&d_output, imageSize));
    
    // Copy input to GPU
    CUDA_CHECK(cudaMemcpy(d_input, input.data.data(), imageSize, cudaMemcpyHostToDevice));
    
    // Configure kernel launch parameters
    dim3 threadsPerBlock(blockSize, blockSize);
    dim3 blocksPerGrid(
        (input.width + blockSize - 1) / blockSize,
        (input.height + blockSize - 1) / blockSize,
        input.channels
    );
    
    // Launch kernel
    if (useSharedMemory) {
        convolutionKernelShared<<<blocksPerGrid, threadsPerBlock>>>(
            d_input, d_output, input.width, input.height, input.channels);
    } else {
        convolutionKernel<<<blocksPerGrid, threadsPerBlock>>>(
            d_input, d_output, input.width, input.height, input.channels);
    }
    
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(output.data.data(), d_output, imageSize, cudaMemcpyDeviceToHost));
    
    // Free GPU memory
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    std::cout << "CUDA convolution completed in: " << duration.count() << " ms" << std::endl;
    
    return output;
}

// Performance analysis
void performanceAnalysis(const Image& input) {
    std::cout << "\n=== Performance Analysis ===" << std::endl;
    
    // Test different block sizes
    std::vector<int> blockSizes = {16, 32};
    std::vector<double> times, timesShared;
    
    for (int blockSize : blockSizes) {
        // Without shared memory
        auto start = std::chrono::high_resolution_clock::now();
        Image result1 = cudaConvolution(input, blockSize, false);
        auto end = std::chrono::high_resolution_clock::now();
        double time1 = std::chrono::duration<double, std::milli>(end - start).count();
        times.push_back(time1);
        
        // With shared memory
        start = std::chrono::high_resolution_clock::now();
        Image result2 = cudaConvolution(input, blockSize, true);
        end = std::chrono::high_resolution_clock::now();
        double time2 = std::chrono::duration<double, std::milli>(end - start).count();
        timesShared.push_back(time2);
        
        std::cout << "Block size " << blockSize << "x" << blockSize << ":" << std::endl;
        std::cout << "  Without shared memory: " << time1 << " ms" << std::endl;
        std::cout << "  With shared memory: " << time2 << " ms" << std::endl;
        std::cout << "  Shared memory speedup: " << time1/time2 << "x" << std::endl;
    }
}

int main() {
    std::cout << "=== CUDA Parallel Convolution Implementation ===" << std::endl;
    
    // Check CUDA device
    int deviceCount;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) {
        std::cerr << "No CUDA devices found!" << std::endl;
        return 1;
    }
    
    cudaDeviceProp deviceProp;
    CUDA_CHECK(cudaGetDeviceProperties(&deviceProp, 0));
    std::cout << "Using GPU: " << deviceProp.name << std::endl;
    std::cout << "Compute capability: " << deviceProp.major << "." << deviceProp.minor << std::endl;
    std::cout << "Global memory: " << deviceProp.totalGlobalMem / (1024*1024) << " MB" << std::endl;
    std::cout << "Shared memory per block: " << deviceProp.sharedMemPerBlock / 1024 << " KB" << std::endl;
    
    // Load test image
    Image inputImg = loadTestImage();
    
    // Perform convolution with different configurations
    Image result32 = cudaConvolution(inputImg, 32, false);
    Image result32Shared = cudaConvolution(inputImg, 32, true);
    
    // Performance analysis
    performanceAnalysis(inputImg);
    
    std::cout << "\nCUDA parallel convolution completed successfully!" << std::endl;
    
    return 0;
}
