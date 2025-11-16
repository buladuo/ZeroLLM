#include "linear.cuh"
#include "matmul.cuh"
#include "transpose.cuh"
#include <iostream>
#include "config.hpp"

/**
 * @brief 添加偏置内核实现
 * 
 * 添加偏置: output += bias
 * @tparam T 数据类型
 * @param output 输出数据 [M, N]
 * @param bias 偏置数据 [N]
 * @param M 行数
 * @param N 列数
 */
template<typename T>
__global__ void add_bias_kernel(T* output, const T* bias, int64_t M, int64_t N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = M * N;
    
    if (idx < total_elements) {
        int row = idx / N;
        int col = idx % N;
        output[idx] += bias[col];
    }
}

/**
 * @brief 添加偏置内核实现
 * 
 * 添加偏置: output += bias
 * @tparam T 数据类型
 * @param output 输出数据 [M, N]
 * @param bias 偏置数据 [N]
 * @param M 行数
 * @param N 列数
 */
template<typename T>
__global__ void compute_weight_grad_kernel(T* d_weight, const T* d_output, const T* input, 
                                          int64_t M, int64_t K, int64_t N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = N * K;
    
    if (idx < total_elements) {
        int out_feat = idx / K;
        int in_feat = idx % K;
        
        T grad = 0;
        for (int64_t i = 0; i < M; i++) {
            grad += d_output[i * N + out_feat] * input[i * K + in_feat];
        }
        d_weight[idx] += grad;
    }
}

// CUDA内核：计算偏置梯度
template<typename T>
__global__ void compute_bias_grad_kernel(T* d_bias, const T* d_output, int64_t M, int64_t N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < N) {
        T grad = 0;
        for (int64_t i = 0; i < M; i++) {
            grad += d_output[i * N + idx];
        }
        d_bias[idx] += grad;
    }
}

template<typename T>
void linear_forward(const T* input,
                        const T* weight,
                        const T* bias,
                        T* output,
                        int64_t M,
                        int64_t K,
                        int64_t N,
                        bool use_bias,
                        cudaStream_t stream) {
    // 线性变换: output = input * weight^T
    // input: [M, K], weight: [N, K] => weight^T: [K, N]
    // output: [M, N]
    matmul_transposed_tiled_A_B_T<T>(input, weight, output, (int)M, (int)N, (int)K, stream);
    
    // 添加偏置
    if (use_bias && bias != nullptr) {
        int total_elements = M * N;
        int block_size = ZEROLLM_DEFAULT_THREADS;
        int grid_size = ZEROLLM_CALC_BLOCKS(total_elements);
        
        add_bias_kernel<T><<<grid_size, block_size, 0, stream>>>(output, bias, M, N);
        CHECK(cudaGetLastError(), "add_bias_kernel launch failed");
    }
}

template<typename T>
void linear_backward(const T* d_output,
                         const T* input,
                         const T* weight,
                         T* d_input,
                         T* d_weight,
                         T* d_bias,
                         int64_t M,
                         int64_t K,
                         int64_t N,
                         bool use_bias,
                         cudaStream_t stream) {
    // 计算输入梯度: d_input = d_output * weight
    // d_output: [M, N], weight: [N, K]
    // d_input: [M, K]
    if (d_input != nullptr) {        
        // 计算 d_input = d_output * weight_transposed
        matmul_tiled<T>(d_output, weight, d_input, (int)M, (int)K, (int)N, stream);
    }
    
    // 计算权重梯度: d_weight += input^T * d_output
    // input: [M, K], d_output: [M, N]
    // d_weight: [K, N] => 转置后为 [N, K]
    if (d_weight != nullptr) {
        // 使用优化的矩阵乘法
        // input^T [K, M] * d_output [M, N] = d_weight [K, N]
        // 但我们存储的是 [N, K] 的转置形式
        matmul_transposed_tiled_A_T_B<T>(input, d_output, d_weight, (int)K, (int)N, (int)M, stream);
    }
    
    // 计算偏置梯度: d_bias += sum(d_output, axis=0)
    // d_output: [M, N]
    // d_bias: [N]
    if (use_bias && d_bias != nullptr) {
        int block_size = ZEROLLM_DEFAULT_THREADS;
        int grid_size = ZEROLLM_CALC_BLOCKS((int)N);
        
        compute_bias_grad_kernel<T><<<grid_size, block_size, 0, stream>>>(d_bias, d_output, M, N);
        CHECK(cudaGetLastError(), "compute_bias_grad_kernel launch failed");
    }
}

// 显式实例化
template void linear_forward<float>(const float* input, 
                                         const float* weight, 
                                         const float* bias,
                                         float* output,
                                         int64_t M, 
                                         int64_t K, 
                                         int64_t N,
                                         bool use_bias,
                                         cudaStream_t stream);

template void linear_backward<float>(const float* d_output,
                                          const float* input,
                                          const float* weight,
                                          float* d_input,
                                          float* d_weight,
                                          float* d_bias,
                                          int64_t M,
                                          int64_t K,
                                          int64_t N,
                                          bool use_bias,
                                          cudaStream_t stream);