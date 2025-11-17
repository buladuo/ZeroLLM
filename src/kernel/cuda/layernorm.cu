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
    
    T sum1 = 0; // 对应 sum(A_i)
    T sum2 = 0; // 对应 sum(A_i * x_hat)

    // 临时存储 A_i 和 x_hat，因为它们需要两次
    // （或者，如果您内存受限，可以只存储 A_i）
    __shared__ T s_A[ZEROLLM_DEFAULT_THREADS]; // 假设 N <= THREADS
    __shared__ T s_x_hat[ZEROLLM_DEFAULT_THREADS]; // 假设 N <= THREADS

    // --- 第一次循环: 计算 A_i 和 x_hat, 并开始归约
    // 注意：如果 N 远大于 blockDim.x，您需要一个临时数组来存储所有的 A_i 和 x_hat
    // 这里我们假设 N 上的循环是主要的（就像您的代码一样）

    T local_sum1 = 0;
    T local_sum2 = 0;

    for (int i = tid; i < N; i += blockDim.x) {
        T x_hat = (x[i] - m) * rs;
        T A_i = dy[i] * gamma[i]; // 这是关键： dL/dx_hat

        local_sum1 += A_i;
        local_sum2 += A_i * x_hat;
    }

    // --- 第一次归约 (Sum1 和 Sum2)
    __shared__ T s_sum1[ZEROLLM_DEFAULT_THREADS];
    __shared__ T s_sum2[ZEROLLM_DEFAULT_THREADS];
    s_sum1[tid] = local_sum1;
    s_sum2[tid] = local_sum2;
    __syncthreads();

    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (tid < offset && tid + offset < blockDim.x) {
            s_sum1[tid] += s_sum1[tid + offset];
            s_sum2[tid] += s_sum2[tid + offset];
        }
        __syncthreads();
    }

    // 只有 thread 0 持有正确的总和
    T total_sum1 = s_sum1[0]; // sum(A_i)
    T total_sum2 = s_sum2[0]; // sum(A_i * x_hat)

    T inv_N = T(1.0) / N;

    // --- 第二次循环: 计算最终的 dx[i]
    for (int i = tid; i < N; i += blockDim.x) {
        T x_hat = (x[i] - m) * rs;
        T A_i = dy[i] * gamma[i];

        // 这是正确的公式:
        // dx[i] = (1/sigma) * [ A_i - (1/N * sum1) - (x_hat / N * sum2) ]
        dx[i] = rs * (A_i - (total_sum1 * inv_N) - (x_hat * total_sum2 * inv_N));
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