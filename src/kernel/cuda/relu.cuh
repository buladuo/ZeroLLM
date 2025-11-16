#ifndef RELU_CUH
#define RELU_CUH

#include <cuda_runtime.h>
#include <cuda.h>

/**
 * @brief Element-wise ReLU操作
 * 
 * @tparam T 数据类型
 * @param input 输入向量
 * @param output 输出向量
 * @param size 向量的长度
 * @param stream 运行时流
 */
template<typename T>
void relu_forward(const T* input, T* output, int size, cudaStream_t stream = 0);


/**
 * @brief Element-wise ReLU反向传播操作
 * 
 * @tparam T 数据类型
 * @param input 前向输入向量
 * @param grad_output 上一层的梯度向量
 * @param grad_input 当前层的梯度向量输出
 * @param size 向量长度
 * @param stream 运行时流
 */
template<typename T>
void relu_backward(const T* input, const T* grad_output, T* grad_input, int size, cudaStream_t stream = 0);



#endif