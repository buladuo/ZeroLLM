#ifndef LAYERNORM_CUH
#define LAYERNORM_CUH

#include <cuda_runtime.h>
#include <cuda.h>

/**
 * @brief LayerNorm前向传播
 * 
 * @tparam T 数据类型
 * @param input 输入张量, shape [M, N]
 * @param gamma 缩放参数, shape [N]
 * @param beta 偏移参数, shape [N]
 * @param output 输出张量, shape [M, N]
 * @param mean 均值缓冲区, shape [M], 可为nullptr
 * @param rstd 标准差倒数缓冲区, shape [M], 可为nullptr
 * @param M 批次大小
 * @param N 特征维度大小
 * @param eps 防止除零的小常数
 * @param stream CUDA流
 */
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
                            cudaStream_t stream = 0);

/**
 * @brief LayerNorm反向传播
 * 
 * @tparam T 数据类型
 * @param input 前向传播时的输入, shape [M, N]
 * @param d_output 输出梯度, shape [M, N]
 * @param gamma 缩放参数, shape [N]
 * @param mean 前向传播时计算的均值, shape [M]
 * @param rstd 前向传播时计算的标准差倒数, shape [M]
 * @param d_input 输入梯度, shape [M, N]
 * @param d_gamma gamma参数梯度, shape [N]
 * @param d_beta beta参数梯度, shape [N]
 * @param M 批次大小
 * @param N 特征维度大小
 * @param stream CUDA流
 */
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
                             cudaStream_t stream = 0);

#endif // LAYERNORM_CUH