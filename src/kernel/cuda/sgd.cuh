#ifndef SGD_CUH
#define SGD_CUH

#include <cuda_runtime.h>
#include <cuda.h>

/**
 * @brief SGD 优化器步进 (SGD Optimizer Step)
 * * 实现公式:
 * 1. g_t = g_t + weight_decay * theta_{t-1} (若 weight_decay != 0)
 * 2. v_t = momentum * v_{t-1} + g_t       (若 momentum != 0)
 * 3. theta_t = theta_{t-1} - lr * v_t
 * * @tparam T 数据类型 (float, double)
 * @param params        模型参数 (theta)，输入并被更新
 * @param grads         梯度 (g)
 * @param velocity      动量缓存 (v)，如果 momentum=0 可为 nullptr
 * @param size          参数总数量
 * @param learning_rate 学习率 (lr)
 * @param momentum      动量系数 (mu)
 * @param weight_decay  权重衰减系数 (lambda)
 * @param stream        CUDA流
 */
template<typename T>
void sgd_step(T* params, 
              const T* grads, 
              T* velocity, 
              int size, 
              float learning_rate, 
              float momentum, 
              float weight_decay, 
              cudaStream_t stream = 0);

#endif // SGD_CUH