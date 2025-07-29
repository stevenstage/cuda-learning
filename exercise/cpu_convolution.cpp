#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <fstream>

// Define Sobel kernels
const int KERNEL_SIZE = 3;
const int SOBEL_X[3][3] = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
const int SOBEL_Y[3][3] = {{-1, -2, -1}, {0, 0, 0}, {1, 2, 1}};

// Simple image structure
struct Image {
    int width, height, channels;
    std::vector<unsigned char> data;
    
    Image(int w, int h, int c) : width(w), height(h), channels(c) {
        data.resize(w * h * c);
    }
    
    // Get pixel value at (x, y) for channel c
    unsigned char getPixel(int x, int y, int c) const {
        if (x < 0 || x >= width || y < 0 || y >= height) {
            return 0; // Zero padding for boundary
        }
        return data[(y * width + x) * channels + c];
    }
    
    // Set pixel value at (x, y) for channel c
    void setPixel(int x, int y, int c, unsigned char value) {
        if (x >= 0 && x < width && y >= 0 && y < height) {
            data[(y * width + x) * channels + c] = value;
        }
    }
};

// Load a simple PPM image (for demonstration - you can replace with actual image loading)
Image loadTestImage() {
    const char* filename = "F1.png";   // ← 换成你的文件名
    int w, h, comp;
    unsigned char* pixels = stbi_load(filename, &w, &h, &comp, 0);
    if (!pixels) {
        std::cerr << "Failed to load " << filename << std::endl;
        exit(1);
    }

    // 若原图是灰度、RGBA 等，统一转成 3 通道
    int channels = (comp >= 3) ? 3 : comp;
    Image img(w, h, channels);

    // 拷贝数据
    std::copy(pixels, pixels + w * h * channels, img.data.begin());
    stbi_image_free(pixels);

    std::cout << "Loaded image: " << w << "x" << h << ", "
              << channels << " channels" << std::endl;
    return img;
}

// CPU Serial Convolution Implementation
Image cpuConvolution(const Image& input) {
    std::cout << "Starting CPU serial convolution..." << std::endl;
    
    auto start = std::chrono::high_resolution_clock::now();
    
    Image output(input.width, input.height, input.channels);
    const int offset = KERNEL_SIZE / 2;
    
    // Process each channel separately
    for (int c = 0; c < input.channels; c++) {
        std::cout << "Processing channel " << (c + 1) << "/" << input.channels << std::endl;
        
        for (int y = 0; y < input.height; y++) {
            for (int x = 0; x < input.width; x++) {
                float sumX = 0.0f, sumY = 0.0f;
                
                // Apply Sobel kernels
                for (int ky = 0; ky < KERNEL_SIZE; ky++) {
                    for (int kx = 0; kx < KERNEL_SIZE; kx++) {
                        int px = x + kx - offset;
                        int py = y + ky - offset;
                        
                        unsigned char pixel = input.getPixel(px, py, c);
                        
                        sumX += pixel * SOBEL_X[ky][kx];
                        sumY += pixel * SOBEL_Y[ky][kx];
                    }
                }
                
                // Calculate magnitude
                float magnitude = sqrt(sumX * sumX + sumY * sumY);
                magnitude = std::min(255.0f, magnitude);
                
                output.setPixel(x, y, c, (unsigned char)magnitude);
            }
        }
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    std::cout << "CPU convolution completed in: " << duration.count() << " ms" << std::endl;
    
    return output;
}

// Save image to file (simple PPM format)
void saveImage(const Image& img, const std::string& filename) {
    std::ofstream file(filename, std::ios::binary);
    
    // PPM header
    file << "P6\n" << img.width << " " << img.height << "\n255\n";
    
    // Write pixel data
    for (int y = 0; y < img.height; y++) {
        for (int x = 0; x < img.width; x++) {
            for (int c = 0; c < img.channels; c++) {
                file << img.getPixel(x, y, c);
            }
        }
    }
    
    file.close();
    std::cout << "Image saved as: " << filename << std::endl;
}

// Compare two images and calculate error
double compareImages(const Image& img1, const Image& img2) {
    if (img1.width != img2.width || img1.height != img2.height || img1.channels != img2.channels) {
        std::cout << "Image dimensions don't match!" << std::endl;
        return -1.0;
    }
    
    double totalError = 0.0;
    int totalPixels = img1.width * img1.height * img1.channels;
    
    for (int i = 0; i < totalPixels; i++) {
        double diff = abs(img1.data[i] - img2.data[i]);
        totalError += diff;
    }
    
    return totalError / totalPixels;
}

int main() {
    std::cout << "=== CPU Serial Convolution Implementation ===" << std::endl;
    
    // Load test image (replace with actual F1.png loading)
    Image inputImg = loadTestImage();
    
    // Perform CPU convolution
    Image outputImg = cpuConvolution(inputImg);
    
    // Save result
    saveImage(outputImg, "cpu_result.ppm");
    
    std::cout << "\nCPU serial convolution completed successfully!" << std::endl;
    std::cout << "Input image: " << inputImg.width << "x" << inputImg.height 
              << " with " << inputImg.channels << " channels" << std::endl;
    
    return 0;
}
