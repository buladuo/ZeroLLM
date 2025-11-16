#include "transpose.cuh"
#include <iostream>
#include "config.hpp"

template<typename T>
__global__ void transpose_kernel(const T* input, T* output, int rows, int cols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = rows * cols;
    
    if (idx < total_elements) {
        int i = idx / cols;  // row
        int j = idx % cols;  // col
        output[j * rows + i] = input[idx];
    }
}

template<typename T>
void transpose(const T* input, T* output, int rows, int cols, cudaStream_t stream) {
    if (rows <= 0 || cols <= 0) return;
    
    int total_elements = rows * cols;
    int block_size = ZEROLLM_DEFAULT_THREADS;
    int grid_size = ZEROLLM_CALC_BLOCKS(total_elements);
    
    transpose_kernel<<<grid_size, block_size, 0, stream>>>(input, output, rows, cols);
    
    CHECK(cudaGetLastError(), "CUDA kernel launch error in transpose()");
}

// 显式实例化常用类型
template void transpose<float>(const float* input, float* output, int rows, int cols, cudaStream_t stream);