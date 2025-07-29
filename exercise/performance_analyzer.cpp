#include <algorithm>   // std::max_element
#include <cuda_runtime.h>  // CUDART_VERSION
#include <iostream>
#include <vector>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <cmath>

struct PerformanceData {
    int blockSize;
    double timeWithoutShared;
    double timeWithShared;
    double speedupCPU;
    double speedupShared;
};

class PerformanceAnalyzer {
private:
    std::vector<PerformanceData> results;
    double cpuTime;
    
public:
    PerformanceAnalyzer(double cpu_time) : cpuTime(cpu_time) {}
    
    void addResult(int blockSize, double timeNoShared, double timeShared) {
        PerformanceData data;
        data.blockSize = blockSize;
        data.timeWithoutShared = timeNoShared;
        data.timeWithShared = timeShared;
        data.speedupCPU = cpuTime / timeNoShared;
        data.speedupShared = timeNoShared / timeShared;
        results.push_back(data);
    }
    
    void generateReport() {
        std::cout << "\n=== DETAILED PERFORMANCE ANALYSIS REPORT ===" << std::endl;
        std::cout << std::string(80, '=') << std::endl;
        
        // System information
        std::cout << "\n1. EXPERIMENTAL ENVIRONMENT" << std::endl;
        std::cout << std::string(40, '-') << std::endl;
        std::cout << "CPU: Intel/AMD Processor (replace with actual model)" << std::endl;
        std::cout << "GPU: NVIDIA GPU (check with nvidia-smi)" << std::endl;
        std::cout << "CUDA Version: " << CUDART_VERSION << std::endl;
        std::cout << "Compiler: nvcc with -O3 optimization" << std::endl;
        std::cout << "Image Size: 1024x1024 pixels, 3 channels" << std::endl;
        std::cout << "Convolution Kernel: 3x3 Sobel filter" << std::endl;
        
        // Algorithm design
        std::cout << "\n2. ALGORITHM DESIGN" << std::endl;
        std::cout << std::string(40, '-') << std::endl;
        std::cout << "CPU Serial Implementation:" << std::endl;
        std::cout << "  - Sequential processing of each pixel" << std::endl;
        std::cout << "  - Zero-padding for boundary handling" << std::endl;
        std::cout << "  - Separate channel processing" << std::endl;
        std::cout << "  - Time Complexity: O(W * H * C * K²)" << std::endl;
        
        std::cout << "\nCUDA Parallel Implementation:" << std::endl;
        std::cout << "  - One thread per pixel approach" << std::endl;
        std::cout << "  - 3D thread grid (width, height, channels)" << std::endl;
        std::cout << "  - Shared memory optimization for data reuse" << std::endl;
        std::cout << "  - Coalesced memory access patterns" << std::endl;
        
        // Performance results
        std::cout << "\n3. PERFORMANCE COMPARISON RESULTS" << std::endl;
        std::cout << std::string(40, '-') << std::endl;
        std::cout << "CPU Serial Time: " << cpuTime << " ms" << std::endl;
        std::cout << std::endl;
        
        // Table header
        std::cout << std::left << std::setw(12) << "Block Size" 
                  << std::setw(15) << "GPU Time (ms)" 
                  << std::setw(18) << "GPU+Shared (ms)"
                  << std::setw(15) << "CPU Speedup"
                  << std::setw(18) << "Shared Speedup" << std::endl;
        std::cout << std::string(78, '-') << std::endl;
        
        for (const auto& data : results) {
            std::cout << std::left << std::setw(12) << (std::to_string(data.blockSize) + "x" + std::to_string(data.blockSize))
                      << std::setw(15) << std::fixed << std::setprecision(2) << data.timeWithoutShared
                      << std::setw(18) << data.timeWithShared
                      << std::setw(15) << data.speedupCPU << "x"
                      << std::setw(18) << data.speedupShared << "x" << std::endl;
        }
        
        // Analysis
        std::cout << "\n4. ANALYSIS AND DISCUSSION" << std::endl;
        std::cout << std::string(40, '-') << std::endl;
        
        // Find best performing configuration
        auto best = std::max_element(results.begin(), results.end(),
            [](const PerformanceData& a, const PerformanceData& b) {
                return a.speedupCPU < b.speedupCPU;
            });
        
        std::cout << "Best CPU Speedup: " << best->speedupCPU << "x with " 
                  << best->blockSize << "x" << best->blockSize << " blocks" << std::endl;
        
        // Shared memory analysis
        double avgSharedSpeedup = 0;
        for (const auto& data : results) {
            avgSharedSpeedup += data.speedupShared;
        }
        avgSharedSpeedup /= results.size();
        
        std::cout << "Average Shared Memory Speedup: " << std::fixed << std::setprecision(2) 
                  << avgSharedSpeedup << "x" << std::endl;
        
        std::cout << "\nKey Observations:" << std::endl;
        std::cout << "• GPU parallelization provides significant speedup over CPU" << std::endl;
        std::cout << "• Shared memory optimization reduces global memory accesses" << std::endl;
        std::cout << "• Block size affects performance due to occupancy and memory coalescing" << std::endl;
        std::cout << "• Larger block sizes may provide better performance for larger images" << std::endl;
        
        std::cout << "\n5. OPTIMIZATION FACTORS" << std::endl;
        std::cout << std::string(40, '-') << std::endl;
        std::cout << "Key factors affecting CUDA convolution performance:" << std::endl;
        std::cout << "• Memory bandwidth utilization" << std::endl;
        std::cout << "• Thread block size and occupancy" << std::endl;
        std::cout << "• Shared memory usage for data reuse" << std::endl;
        std::cout << "• Memory coalescing patterns" << std::endl;
        std::cout << "• Computation to memory access ratio" << std::endl;
        
        std::cout << "\n6. FUTURE IMPROVEMENTS" << std::endl;
        std::cout << std::string(40, '-') << std::endl;
        std::cout << "Potential optimizations:" << std::endl;
        std::cout << "• Texture memory for better cache locality" << std::endl;
        std::cout << "• Multiple images processing in batches" << std::endl;
        std::cout << "• Separable convolution for larger kernels" << std::endl;
        std::cout << "• Warp-level primitives for better utilization" << std::endl;
        std::cout << "• Mixed precision computation" << std::endl;
        
        std::cout << std::string(80, '=') << std::endl;
    }
    
    void saveCSVReport(const std::string& filename) {
        std::ofstream file(filename);
        file << "Block_Size,GPU_Time_ms,GPU_Shared_Time_ms,CPU_Speedup,Shared_Speedup\n";
        
        for (const auto& data : results) {
            file << data.blockSize << ","
                 << data.timeWithoutShared << ","
                 << data.timeWithShared << ","
                 << data.speedupCPU << ","
                 << data.speedupShared << "\n";
        }
        
        file.close();
        std::cout << "\nPerformance data saved to: " << filename << std::endl;
    }
    
    void generatePlotScript() {
        std::ofstream file("plot_performance.py");
        file << R"(import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Read performance data
df = pd.read_csv('performance_results.csv')

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

# Plot 1: Execution time comparison
block_sizes = [f"{x}x{x}" for x in df['Block_Size']]
ax1.bar(np.arange(len(block_sizes)) - 0.2, df['GPU_Time_ms'], 0.4, label='GPU without Shared')
ax1.bar(np.arange(len(block_sizes)) + 0.2, df['GPU_Shared_Time_ms'], 0.4, label='GPU with Shared')
ax1.set_xlabel('Block Size')
ax1.set_ylabel('Execution Time (ms)')
ax1.set_title('GPU Execution Time Comparison')
ax1.set_xticks(np.arange(len(block_sizes)))
ax1.set_xticklabels(block_sizes)
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: CPU Speedup
ax2.plot(df['Block_Size'], df['CPU_Speedup'], 'bo-', linewidth=2, markersize=8)
ax2.set_xlabel('Block Size')
ax2.set_ylabel('Speedup Factor')
ax2.set_title('CPU to GPU Speedup')
ax2.grid(True, alpha=0.3)

# Plot 3: Shared Memory Speedup
ax3.bar(block_sizes, df['Shared_Speedup'], color='green', alpha=0.7)
ax3.set_xlabel('Block Size')
ax3.set_ylabel('Speedup Factor')
ax3.set_title('Shared Memory Speedup')
ax3.grid(True, alpha=0.3)

# Plot 4: Performance Summary
categories = ['16x16', '32x32']
gpu_times = df['GPU_Time_ms'].tolist()
shared_times = df['GPU_Shared_Time_ms'].tolist()

x = np.arange(len(categories))
width = 0.35

bars1 = ax4.bar(x - width/2, gpu_times, width, label='GPU Global Memory')
bars2 = ax4.bar(x + width/2, shared_times, width, label='GPU Shared Memory')

ax4.set_xlabel('Thread Block Configuration')
ax4.set_ylabel('Execution Time (ms)')
ax4.set_title('Performance Summary')
ax4.set_xticks(x)
ax4.set_xticklabels(categories)
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('performance_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

print("Performance analysis plots saved as 'performance_analysis.png'")
)";
        file.close();
        std::cout << "Python plotting script generated: plot_performance.py" << std::endl;
        std::cout << "Run with: python plot_performance.py" << std::endl;
    }
};

// Usage example (to be integrated with main CUDA program)

int main() {
    // Simulate performance data (replace with actual measurements)
    double cpuTime = 1500.0; // ms
    
    PerformanceAnalyzer analyzer(cpuTime);
    
    // Add results from different configurations
    analyzer.addResult(16, 45.2, 38.7);
    analyzer.addResult(32, 42.1, 35.9);
    
    // Generate comprehensive report
    analyzer.generateReport();
    analyzer.saveCSVReport("performance_results.csv");
    analyzer.generatePlotScript();
    
    return 0;
}
