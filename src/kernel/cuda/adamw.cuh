#ifndef ADAMW_CUH
#define ADAMW_CUH

#include <cuda_runtime.h>
#include <cuda.h>

/**
 * @brief AdamW 优化器步进 (AdamW Optimizer Step)
 * * 实现逻辑:
 * 1. Weight Decay: p_t = p_{t-1} - lr * lambda * p_{t-1}
 * 2. Adam Update:  p_t = p_t - lr * m_hat / (sqrt(v_hat) + eps)
 * (两者在 Kernel 中合并为一次写入)
 * * @tparam T 数据类型 (float)
 * @param params           模型参数 (theta)
 * @param grads            梯度 (g)
 * @param m                一阶矩估计 (m)
 * @param v                二阶矩估计 (v)
 * @param size             参数总数量
 * @param learning_rate    学习率 (lr)
 * @param weight_decay     权重衰减系数 (lambda)
 * @param beta1            一阶矩衰减率
 * @param beta2            二阶矩衰减率
 * @param eps              数值稳定项 (epsilon)
 * @param bias_correction1 预计算的偏差修正项1 (1 - beta1^t)
 * @param bias_correction2 预计算的偏差修正项2 (1 - beta2^t)
 * @param stream           CUDA流
 */
template<typename T>
void adamw_step(T* params, 
                const T* grads, 
                T* m, 
                T* v, 
                int size, 
                float learning_rate, 
                float weight_decay,
                float beta1, 
                float beta2, 
                float eps, 
                float bias_correction1,
                float bias_correction2,
                cudaStream_t stream = 0);

#endif // ADAMW_CUH