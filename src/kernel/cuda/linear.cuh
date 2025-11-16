#ifndef LINEAR_CUH
#define LINEAR_CUH

#include <cuda_runtime.h>
#include <cuda.h>

/**
 * @brief 线性层前向传播: Y = X * W^T + b
 * 
 * @tparam T 数据类型
 * @param input 输入张量, shape [M, K]
 * @param weight 权重矩阵, shape [N, K]
 * @param bias 偏置向量, shape [N]
 * @param output 输出张量, shape [M, N]
 * @param M 输入的批次大小
 * @param K 输入特征数
 * @param N 输出特征数
 * @param use_bias 是否使用偏置
 * @param stream CUDA流
 */
template<typename T>
void linear_forward(const T* input, 
                         const T* weight, 
                         const T* bias,
                         T* output,
                         int64_t M, 
                         int64_t K, 
                         int64_t N,
                         bool use_bias,
                         cudaStream_t stream = 0);

/**
 * @brief 线性层反向传播
 * 
 * @tparam T 数据类型
 * @param d_output 输出梯度, shape [M, N]
 * @param input 前向传播时的输入, shape [M, K]
 * @param weight 权重矩阵, shape [N, K]
 * @param d_input 输入梯度, shape [M, K]
 * @param d_weight 权重梯度, shape [N, K]
 * @param d_bias 偏置梯度, shape [N]
 * @param M 输入的批次大小
 * @param K 输入特征数
 * @param N 输出特征数
 * @param use_bias 是否使用偏置
 * @param stream CUDA流
 */
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
                          cudaStream_t stream = 0);

#endif // LINEAR_CUH