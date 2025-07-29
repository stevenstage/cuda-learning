# Makefile for CPU and CUDA Convolution Implementations
# Assignment 2: Image Convolution Performance Comparison

# Compiler settings
CXX = g++
NVCC = nvcc
CXXFLAGS = -O3 -std=c++11 -Wall
NVCCFLAGS = -O3 -std=c++11 -arch=sm_35
LDFLAGS = -lm

# Source files
CPU_SRC = cpu_convolution.cpp
CUDA_SRC = cuda_convolution.cu

# Output executables
CPU_EXEC = cpu_convolution
CUDA_EXEC = cuda_convolution

# Default target
all: check_cuda $(CPU_EXEC) $(CUDA_EXEC)

# Check if CUDA is available
check_cuda:
	@which nvcc > /dev/null 2>&1 || (echo "CUDA not found. Installing CPU version only." && $(MAKE) cpu_only)

# CPU only build (fallback when CUDA is not available)
cpu_only: $(CPU_EXEC)
	@echo "CPU convolution compiled successfully!"
	@echo "Run with: ./$(CPU_EXEC)"

# Build CPU version
$(CPU_EXEC): $(CPU_SRC)
	@echo "Compiling CPU serial convolution..."
	$(CXX) $(CXXFLAGS) $(CPU_SRC) -o $(CPU_EXEC) $(LDFLAGS)
	@echo "CPU compilation completed."

# Build CUDA version
$(CUDA_EXEC): $(CUDA_SRC)
	@echo "Compiling CUDA parallel convolution..."
	$(NVCC) $(NVCCFLAGS) $(CUDA_SRC) -o $(CUDA_EXEC)
	@echo "CUDA compilation completed."

# Run performance comparison
test: all
	@echo "=== Running Performance Tests ==="
	@echo ""
	@echo "1. Running CPU serial convolution..."
	@./$(CPU_EXEC) | tee cpu_output.log
	@echo ""
	@echo "2. Running CUDA parallel convolution..."
	@./$(CUDA_EXEC) | tee cuda_output.log
	@echo ""
	@echo "=== Test Results ==="
	@echo "CPU results logged to: cpu_output.log"
	@echo "CUDA results logged to: cuda_output.log"

# Run only CPU version
test_cpu: $(CPU_EXEC)
	@echo "Running CPU convolution test..."
	@./$(CPU_EXEC)

# Run only CUDA version
test_cuda: $(CUDA_EXEC)
	@echo "Running CUDA convolution test..."
	@./$(CUDA_EXEC)

# Generate performance report
report: test
	@echo "=== Generating Performance Report ==="
	@python3 -c "import sys; print('Python available for plotting')" 2>/dev/null || echo "Python3 not available for plotting"
	@echo "Performance data ready for analysis"

# Clean build files
clean:
	@echo "Cleaning build files..."
	rm -f $(CPU_EXEC) $(CUDA_EXEC)
	rm -f *.log *.txt *.csv *.png
	rm -f cpu_result.ppm
	@echo "Clean completed."

# Install dependencies (Ubuntu/Debian)
install_deps:
	@echo "Installing dependencies..."
	sudo apt-get update
	sudo apt-get install -y build-essential python3 python3-matplotlib python3-pandas python3-numpy
	@echo "Dependencies installed."

# Display system information
sysinfo:
	@echo "=== System Information ==="
	@echo "CPU Info:"
	@cat /proc/cpuinfo | grep "model name" | head -1
	@echo ""
	@echo "Memory Info:"
	@free -h | head -2
	@echo ""
	@echo "CUDA Info:"
	@nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader,nounits 2>/dev/null || echo "NVIDIA GPU not detected"
	@echo ""
	@echo "CUDA Version:"
	@nvcc --version 2>/dev/null || echo "CUDA not installed"

# Help target
help:
	@echo "=== Available Make Targets ==="
	@echo ""
	@echo "Build targets:"
	@echo "  all         - Build both CPU and CUDA versions"
	@echo "  cpu_only    - Build only CPU version"
	@echo "  $(CPU_EXEC) - Build CPU serial implementation"
	@echo "  $(CUDA_EXEC) - Build CUDA parallel implementation"
	@echo ""
	@echo "Test targets:"
	@echo "  test        - Run both CPU and CUDA performance tests"
	@echo "  test_cpu    - Run only CPU test"
	@echo "  test_cuda   - Run only CUDA test"
	@echo "  report      - Generate performance analysis report"
	@echo ""
	@echo "Utility targets:"
	@echo "  clean       - Remove all build files and outputs"
	@echo "  sysinfo     - Display system information"
	@echo "  install_deps- Install required dependencies (Ubuntu/Debian)"
	@echo "  help        - Show this help message"
	@echo ""
	@echo "Usage examples:"
	@echo "  make all       # Build everything"
	@echo "  make test      # Run performance comparison"
	@echo "  make clean     # Clean up files"
	@echo "  make sysinfo   # Check system specs"

# Prevent make from treating these as files
.PHONY: all cpu_only test test_cpu test_cuda report clean install_deps sysinfo help check_cuda
