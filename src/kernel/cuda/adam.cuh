#ifndef ADAM_CUH
#define ADAM_CUH

#include <cuda_runtime.h>
#include <cuda.h>

/**
 * @brief Adam 优化器步进 (Adam Optimizer Step)
 * * @tparam T 数据类型 (float)
 * @param params           模型参数 (theta)，输入并被更新
 * @param grads            梯度 (g)
 * @param m                一阶矩估计 (m)
 * @param v                二阶矩估计 (v)
 * @param size             参数总数量
 * @param learning_rate    学习率 (alpha)
 * @param beta1            一阶矩衰减率
 * @param beta2            二阶矩衰减率
 * @param eps              数值稳定项 (epsilon)
 * @param bias_correction1 预计算的偏差修正项1 (1 - beta1^t)
 * @param bias_correction2 预计算的偏差修正项2 (1 - beta2^t)
 * @param stream           CUDA流
 */
template<typename T>
void adam_step(T* params, 
               const T* grads, 
               T* m, 
               T* v, 
               int size, 
               float learning_rate, 
               float beta1, 
               float beta2, 
               float eps, 
               float bias_correction1,
               float bias_correction2,
               cudaStream_t stream = 0);

#endif // ADAM_CUH