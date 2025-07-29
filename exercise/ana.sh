#!/bin/bash

echo "=== Compilation and Performance Analysis Script ==="
echo ""

# Check if CUDA is available
if ! command -v nvcc &> /dev/null; then
    echo "CUDA compiler (nvcc) not found. Please install CUDA toolkit."
    echo "Compiling CPU version only..."
    
    # Compile CPU version
    echo "Compiling CPU serial convolution..."
    g++ -O3 -std=c++11 cpu_convolution.cpp -o cpu_convolution
    
    if [ $? -eq 0 ]; then
        echo "CPU compilation successful!"
        echo "Running CPU convolution..."
        ./cpu_convolution
    else
        echo "CPU compilation failed!"
        exit 1
    fi
    
    exit 0
fi

echo "CUDA compiler found. Compiling both CPU and CUDA versions..."

# Compile CPU version
echo "1. Compiling CPU serial convolution..."
g++ -O3 -std=c++11 cpu_convolution.cpp -o cpu_convolution

if [ $? -ne 0 ]; then
    echo "CPU compilation failed!"
    exit 1
fi

# Compile CUDA version
echo "2. Compiling CUDA parallel convolution..."
nvcc -O3 -std=c++11 cuda_convolution.cu -o cuda_convolution

if [ $? -ne 0 ]; then
    echo "CUDA compilation failed!"
    exit 1
fi

echo "Both compilations successful!"
echo ""

# Run performance comparison
echo "=== Running Performance Comparison ==="
echo ""

echo "Running CPU serial convolution..."
./cpu_convolution > cpu_results.txt 2>&1
CPU_TIME=$(grep "completed in:" cpu_results.txt | grep -o '[0-9]\+' | head -1)

echo "Running CUDA parallel convolution..."
./cuda_convolution > cuda_results.txt 2>&1
CUDA_TIME=$(grep "completed in:" cuda_results.txt | grep -o '[0-9]\+' | head -1)

echo ""
echo "=== Performance Results ==="
if [ ! -z "$CPU_TIME" ] && [ ! -z "$CUDA_TIME" ]; then
    echo "CPU Time: ${CPU_TIME} ms"
    echo "CUDA Time: ${CUDA_TIME} ms"
    
    # Calculate speedup (using bc for floating point arithmetic)
    if command -v bc &> /dev/null; then
        SPEEDUP=$(echo "scale=2; $CPU_TIME / $CUDA_TIME" | bc)
        echo "Speedup: ${SPEEDUP}x"
    else
        echo "Install 'bc' calculator for speedup calculation"
    fi
else
    echo "Could not extract timing information. Check output files."
fi

echo ""
echo "Detailed results saved in:"
echo "- cpu_results.txt"
echo "- cuda_results.txt"
echo ""
echo "Generated output images:"
echo "- cpu_result.ppm (CPU convolution result)"
