#include "softmax.cuh"
#include "config.hpp"
#include <iostream>
#include <float.h> // for FLT_MAX

/**
 * @brief 逐行Softmax Kernel
 * 每个block处理输入矩阵的一行。
 * 数值稳定: 减去该行最大值后再进行exp计算。
 */
template<typename T>
__global__ void softmax_kernel(const T* __restrict__ input,
                               T* __restrict__ output,
                               int M, int N) {
    extern __shared__ T shared[];

    int row = blockIdx.x;
    if (row >= M) return;

    const T* row_input = input + row * N;
    T* row_output = output + row * N;

    // 每个线程处理多个元素（循环展开）
    T local_max = -FLT_MAX;
    for (int i = threadIdx.x; i < N; i += blockDim.x) {
        local_max = fmaxf(local_max, row_input[i]);
    }

    // 计算全局最大值
    shared[threadIdx.x] = local_max;
    __syncthreads();

    // block内归约最大值
    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (threadIdx.x < offset) {
            shared[threadIdx.x] = fmaxf(shared[threadIdx.x], shared[threadIdx.x + offset]);
        }
        __syncthreads();
    }
    T row_max = shared[0];

    // 计算 exp(x - max)
    T local_sum = 0;
    for (int i = threadIdx.x; i < N; i += blockDim.x) {
        T val = expf(row_input[i] - row_max);
        row_output[i] = val;
        local_sum += val;
    }

    // 归约求和
    shared[threadIdx.x] = local_sum;
    __syncthreads();
    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (threadIdx.x < offset) {
            shared[threadIdx.x] += shared[threadIdx.x + offset];
        }
        __syncthreads();
    }
    T row_sum = shared[0];

    // 归一化
    for (int i = threadIdx.x; i < N; i += blockDim.x) {
        row_output[i] /= row_sum;
    }
}


template<typename T>
void softmax(const T* input, T* output, int M, int N, cudaStream_t stream) {
    if (M <= 0 || N <= 0)
        throw std::invalid_argument("softmax: M and N must be > 0");

    const int THREADS = ZEROLLM_DEFAULT_THREADS;
    dim3 grid(M);
    dim3 block(THREADS);
    size_t shmem = THREADS * sizeof(T);

    softmax_kernel<<<grid, block, shmem, stream>>>(input, output, M, N);
    CHECK(cudaGetLastError(), "softmax_kernel launch failed");
}


// 显式实例化
template void softmax<float>(const float* input, float* output, int M, int N, cudaStream_t stream);
