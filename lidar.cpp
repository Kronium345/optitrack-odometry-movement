#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <limits>
#ifdef __CUDACC__
#include <cuda_runtime.h>
#define CUDA_AVAILABLE true
#else
#define CUDA_AVAILABLE false
#endif

// Function to process LIDAR data
std::vector<float> lidar_process2(const std::vector<std::tuple<float, float, float>>& lidar_data, float vertical_limit, int segments = 72, float maximum = 6.0, bool norm = true) {
    int i = 0;
    std::vector<float> segmented(segments, maximum);
    
    for (const auto& point : lidar_data) {
        float x = std::get<0>(point);
        float y = std::get<1>(point);
        float z = std::get<2>(point);
        
        if (z > vertical_limit || z < -vertical_limit) {
            continue;
        }
        i += 1;
        if (i % 4 != 0) {
            continue;
        }
        
        float dis = std::sqrt(x * x + y * y);
        float ang = std::atan2(y, x) * 180.0 / M_PI;
        int bucket = static_cast<int>(std::floor(ang / (360.0 / segments)));
        if ((x < 0 && y < 0) || (x < 0 && y > 0)) {
            bucket += segments / 2;
        }
        bucket = bucket % segments;
        segmented[bucket] = std::min(segmented[bucket], dis);
    }
    
    if (norm) {
        for (auto& val : segmented) {
            val = std::clamp(val / maximum, 0.0f, 1.0f);
        }
    }
    
    return segmented;
}

int main() {
    std::vector<std::tuple<float, float, float>> lidar_data = {
        // Example data: (x, y, z)
        {1.0, 2.0, 0.5},
        {2.0, 1.0, 0.2},
    };
    float vertical_limit = 1.0;
    int segments = 72;
    float maximum = 6.0;
    bool norm = true;
    
    std::vector<float> lidar = lidar_process2(lidar_data, vertical_limit, segments, maximum, norm);
    
    // Displaying the processed lidar data
    std::cout << "Processed LIDAR data: ";
    for (const auto& val : lidar) {
        std::cout << val << " ";
    }
    std::cout << std::endl;
    
    return 0;
}
