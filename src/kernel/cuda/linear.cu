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
        // C[M, K] = A[M, N] * B[N, K]
        // 映射到 matmul_tiled(A, B, C, M, N, K)
        // A = d_output, B = weight, C = d_input
        // M (C_rows) = M
        // N (C_cols) = K
        // K (Reduce) = N
        matmul_tiled<T>(d_output, weight, d_input, (int)M, (int)K, (int)N, stream);
        // (此部分调用在你的原始代码中是正确的)
    }
    
    // 计算权重梯度: d_weight = d_output^T * input
    // d_output (A): [M, N]
    // input    (B): [M, K]
    // d_weight (C): [N, K]
    if (d_weight != nullptr) {
        // *** 这是修正的部分 ***
        //
        // 目标: d_weight[N, K] = d_output^T[N, M] * input[M, K]
        //
        // 映射到 matmul_transposed_tiled_A_T_B(A, B, C, M_reduce, N_out, K_out)
        // A (kernel) = d_output, 物理 [M, N]
        // B (kernel) = input,    物理 [M, K]
        // C (kernel) = d_weight, 物理 [N, K]
        //
        // 内核布局期望:
        // A [M_reduce, K_out] => [M, N]
        // B [M_reduce, N_out] => [M, K]
        // C [K_out, N_out]    => [N, K]
        //
        // 维度映射:
        // M_reduce (归约) = M
        // K_out    (C行)  = N
        // N_out    (C列)  = K
        //
        // 原始调用 (错误): matmul_..._A_T_B(input, d_output, d_weight, (int)K, (int)N, (int)M, stream);
        //
        // 正确调用:
        matmul_transposed_tiled_A_T_B<T>(
            d_output, // A
            input,    // B
            d_weight, // C
            (int)M,   // M_reduce
            (int)K,   // N_out (C的列数)
            (int)N,   // K_out (C的行数)
            stream
        );
    }
    
    // 计算偏置梯度: d_bias += sum(d_output, axis=0)
    // (此部分调用在你的原始代码中是正确的)
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