#include "relu.cuh"
#include <iostream>
#include "config.hpp"

template<typename T>
__global__ void relu_forward_kernel(const T* input, T* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = input[idx] > 0 ? input[idx] : 0;
    }
}

template<typename T>
__global__ void relu_backward_kernel(const T* input, const T* grad_output, T* grad_input, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        grad_input[idx] = input[idx] > 0 ? grad_output[idx] : 0;
    }
}

template<typename T>
void relu_forward(const T* input, T* output, int size, cudaStream_t stream) {
    if (size <= 0) return;
    
    int block_size = ZEROLLM_DEFAULT_THREADS;
    int grid_size = ZEROLLM_CALC_BLOCKS(size);
    
    relu_forward_kernel<<<grid_size, block_size, 0, stream>>>(input, output, size);
    
    CHECK(cudaGetLastError(), "CUDA kernel launch error in relu_forward()");
}

template<typename T>
void relu_backward(const T* input, const T* grad_output, T* grad_input, int size, cudaStream_t stream) {
    if (size <= 0) return;
    
    int block_size = ZEROLLM_DEFAULT_THREADS;
    int grid_size = ZEROLLM_CALC_BLOCKS(size);
    
    relu_backward_kernel<<<grid_size, block_size, 0, stream>>>(input, grad_output, grad_input, size);
    
    CHECK(cudaGetLastError(), "CUDA kernel launch error in relu_backward()");
}

// 显式实例化常用类型
template void relu_forward<float>(const float* input, float* output, int size, cudaStream_t stream);

template void relu_backward<float>(const float* input, const float* grad_output, float* grad_input, int size, cudaStream_t stream);