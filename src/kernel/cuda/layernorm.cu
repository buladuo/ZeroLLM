#include "layernorm.cuh"
#include <iostream>
#include "config.hpp"

template<typename T>
__global__ void layernorm_forward_kernel(const T* input,
                                        const T* gamma,
                                        const T* beta,
                                        T* output,
                                        T* mean,
                                        T* rstd,
                                        int64_t M,
                                        int64_t N,
                                        T eps) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    
    if (row >= M) return;
    
    const T* x = input + row * N;
    T* y = output + row * N;
    
    // 计算均值
    T sum = 0;
    for (int i = tid; i < N; i += blockDim.x) {
        sum += x[i];
    }
    
    // 使用归约算法计算总和
    __shared__ T s_sum[ZEROLLM_DEFAULT_THREADS];  // 假设最大 256 线程
    s_sum[tid] = sum;
    __syncthreads();
    
    // 归约求和
    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (tid < offset && tid + offset < blockDim.x) {
            s_sum[tid] += s_sum[tid + offset];
        }
        __syncthreads();
    }
    
    T m = s_sum[0] / N;
    if (tid == 0 && mean != nullptr) {
        mean[row] = m;
    }
    
    // 计算方差
    sum = 0;
    for (int i = tid; i < N; i += blockDim.x) {
        T diff = x[i] - m;
        sum += diff * diff;
    }
    
    // 使用归约算法计算方差总和
    s_sum[tid] = sum;
    __syncthreads();
    
    // 归约求和
    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (tid < offset && tid + offset < blockDim.x) {
            s_sum[tid] += s_sum[tid + offset];
        }
        __syncthreads();
    }
    
    T variance = s_sum[0] / N;
    T rs = rsqrtf(variance + eps);
    if (tid == 0 && rstd != nullptr) {
        rstd[row] = rs;
    }
    
    // 归一化
    for (int i = tid; i < N; i += blockDim.x) {
        y[i] = (x[i] - m) * rs * gamma[i] + beta[i];
    }
}

template<typename T>
__global__ void layernorm_backward_kernel(const T* input,
                                         const T* d_output,
                                         const T* gamma,
                                         const T* mean,
                                         const T* rstd,
                                         T* d_input,
                                         T* d_gamma,
                                         T* d_beta,
                                         int64_t M,
                                         int64_t N) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    
    if (row >= M) return;
    
    const T* x = input + row * N;
    const T* dy = d_output + row * N;
    T* dx = d_input + row * N;
    
    T m = mean[row];
    T rs = rstd[row];
    
    // 计算 d_gamma 和 d_beta 的梯度
    // 对每个特征维度分别计算并累加（每个 block 处理一个样本，需要累加所有样本的梯度）
    for (int i = tid; i < N; i += blockDim.x) {
        T x_hat = (x[i] - m) * rs;
        // d_gamma[i] = sum over all samples: dy[i] * x_hat
        atomicAdd(&d_gamma[i], dy[i] * x_hat);
        // d_beta[i] = sum over all samples: dy[i]
        atomicAdd(&d_beta[i], dy[i]);
    }
    __syncthreads();
    
    // 计算 d_input
    T sum_dy = 0;
    T sum_dy_x = 0;
    for (int i = tid; i < N; i += blockDim.x) {
        sum_dy += dy[i];
        sum_dy_x += dy[i] * (x[i] - m);
    }
    
    // 使用归约算法计算总和
    __shared__ T s_sum_dy[256], s_sum_dy_x[256];
    s_sum_dy[tid] = sum_dy;
    s_sum_dy_x[tid] = sum_dy_x;
    __syncthreads();
    
    // 归约求和
    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (tid < offset && tid + offset < blockDim.x) {
            s_sum_dy[tid] += s_sum_dy[tid + offset];
            s_sum_dy_x[tid] += s_sum_dy_x[tid + offset];
        }
        __syncthreads();
    }
    
    T inv_N = T(1.0) / N;
    for (int i = tid; i < N; i += blockDim.x) {
        T x_hat = (x[i] - m) * rs;
        dx[i] = (dy[i] - (s_sum_dy[0] * inv_N) - x_hat * (s_sum_dy_x[0] * inv_N)) * rs * gamma[i];
    }
}

template<typename T>
void cuda_layernorm_forward(const T* input,
                           const T* gamma,
                           const T* beta,
                           T* output,
                           T* mean,
                           T* rstd,
                           int64_t M,
                           int64_t N,
                           T eps,
                           cudaStream_t stream) {
    if (M <= 0 || N <= 0) return;
    
    int block_size = min((int)N, ZEROLLM_DEFAULT_THREADS);
    int grid_size = M;
    
    layernorm_forward_kernel<<<grid_size, block_size, 0, stream>>>(
        input, gamma, beta, output, mean, rstd, M, N, eps);
    CHECK(cudaGetLastError(), "layernorm_forward_kernel launch failed");
}

template<typename T>
void cuda_layernorm_backward(const T* input,
                            const T* d_output,
                            const T* gamma,
                            const T* mean,
                            const T* rstd,
                            T* d_input,
                            T* d_gamma,
                            T* d_beta,
                            int64_t M,
                            int64_t N,
                            cudaStream_t stream) {
    if (M <= 0 || N <= 0) return;
    
    int block_size = min((int)N, ZEROLLM_DEFAULT_THREADS);
    int grid_size = M;
    
    layernorm_backward_kernel<<<grid_size, block_size, 0, stream>>>(
        input, d_output, gamma, mean, rstd, d_input, d_gamma, d_beta, M, N);
    CHECK(cudaGetLastError(), "layernorm_backward_kernel launch failed");
}

// 显式实例化
template void cuda_layernorm_forward<float>(const float* input,
                                           const float* gamma,
                                           const float* beta,
                                           float* output,
                                           float* mean,
                                           float* rstd,
                                           int64_t M,
                                           int64_t N,
                                           float eps,
                                           cudaStream_t stream);

template void cuda_layernorm_backward<float>(const float* input,
                                            const float* d_output,
                                            const float* gamma,
                                            const float* mean,
                                            const float* rstd,
                                            float* d_input,
                                            float* d_gamma,
                                            float* d_beta,
                                            int64_t M,
                                            int64_t N,
                                            cudaStream_t stream);