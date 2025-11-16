#include "add.cuh"
#include <iostream>
#include "config.hpp"  // 复用你的 CHECK 宏等定义

/**
 * @brief 元素级相加 Kernel: output = a + b
 */
template<typename T>
__global__ void add_kernel(const T* __restrict__ a,
                           const T* __restrict__ b,
                           T* __restrict__ output,
                           int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = a[idx] + b[idx];
    }
}


/**
 * @brief 元素级相加 Inplace Kernel: a = a + b
 */
template<typename T>
__global__ void add_inplace_kernel(T* __restrict__ a,
                                   const T* __restrict__ b,
                                   int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        a[idx] += b[idx];
    }
}


template<typename T>
void add(const T* a, const T* b, T* output, int size, cudaStream_t stream) {
    if (size <= 0) {
        throw std::invalid_argument("add: size must be > 0");
    }

    const int BLOCK_SIZE = ZEROLLM_DEFAULT_THREADS;
    const int GRID_SIZE = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;

    add_kernel<<<GRID_SIZE, BLOCK_SIZE, 0, stream>>>(a, b, output, size);
    CHECK(cudaGetLastError(), "add_kernel launch failed");
}


template<typename T>
void add_inplace(T* a, const T* b, int size, cudaStream_t stream) {
    if (size <= 0) {
        throw std::invalid_argument("add_inplace: size must be > 0");
    }

    const int BLOCK_SIZE = ZEROLLM_DEFAULT_THREADS;
    const int GRID_SIZE = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;

    add_inplace_kernel<<<GRID_SIZE, BLOCK_SIZE, 0, stream>>>(a, b, size);
    CHECK(cudaGetLastError(), "add_inplace_kernel launch failed");
}


// 显式实例化
template void add<float>(const float* a, const float* b, float* output, int size, cudaStream_t stream);
template void add_inplace<float>(float* a, const float* b, int size, cudaStream_t stream);
