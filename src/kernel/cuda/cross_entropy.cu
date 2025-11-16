#include "cross_entropy.cuh"
#include "../config.hpp"
#include <cmath>

__global__ void cross_entropy_backward_kernel_impl(float* d_logits,
                                                   const float* softmax_output,
                                                   const int* targets,
                                                   int batch_size,
                                                   int num_classes) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * num_classes;
    
    if (idx >= total_elements) return;
    
    int batch_idx = idx / num_classes;
    int class_idx = idx % num_classes;
    
    int target = targets[batch_idx];
    if (target >= 0 && target < num_classes) {
        if (class_idx == target) {
            d_logits[idx] = softmax_output[idx] - 1.0f;
        } else {
            d_logits[idx] = softmax_output[idx];
        }
    } else {
        d_logits[idx] = softmax_output[idx];
    }
    
    // 归一化梯度
    d_logits[idx] /= batch_size;
}

void cross_entropy_backward_kernel(float* d_logits,
                                   const float* softmax_output,
                                   const int* targets,
                                   int batch_size,
                                   int num_classes,
                                   cudaStream_t stream) {
    if (batch_size <= 0 || num_classes <= 0) return;
    
    int total_elements = batch_size * num_classes;
    int block_size = ZEROLLM_DEFAULT_THREADS;
    int grid_size = (total_elements + block_size - 1) / block_size;
    
    cross_entropy_backward_kernel_impl<<<grid_size, block_size, 0, stream>>>(
        d_logits, softmax_output, targets, batch_size, num_classes);
    
    zerollm_backend::check_last_error("cross_entropy_backward_kernel failed");
}