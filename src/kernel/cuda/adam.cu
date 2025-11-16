#include "adam.cuh"
#include <iostream>
#include "config.hpp"
/**
 * @brief Adam Kernel
 * 实现逻辑完全对应提供的 C++ 代码：
 * 1. Update m: m_t = beta1 * m_{t-1} + (1 - beta1) * g
 * 2. Update v: v_t = beta2 * v_{t-1} + (1 - beta2) * g^2
 * 3. Correction: m_hat = m_t / bc1, v_hat = v_t / bc2
 * 4. Update param: p_t = p_{t-1} - lr * m_hat / (sqrt(v_hat) + eps)
 */
template<typename T>
__global__ void adam_kernel(T* __restrict__ params,
                            const T* __restrict__ grads,
                            T* __restrict__ m,
                            T* __restrict__ v,
                            int size,
                            float learning_rate,
                            float beta1,
                            float beta2,
                            float eps,
                            float bias_correction1,
                            float bias_correction2) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        // 读取全局内存到寄存器
        T g_val = grads[idx];
        T m_val = m[idx];
        T v_val = v[idx];
        T p_val = params[idx];

        // 1. 更新一阶矩估计 (Momentum)
        // m = beta1 * m + (1 - beta1) * g
        m_val = static_cast<T>(beta1) * m_val + static_cast<T>(1.0f - beta1) * g_val;

        // 2. 更新二阶矩估计 (RMSProp)
        // v = beta2 * v + (1 - beta2) * g * g
        v_val = static_cast<T>(beta2) * v_val + static_cast<T>(1.0f - beta2) * g_val * g_val;

        // 写回 m 和 v (供下一次迭代使用)
        m[idx] = m_val;
        v[idx] = v_val;

        // 3. 计算偏差修正后的估计量
        T m_hat = m_val / static_cast<T>(bias_correction1);
        T v_hat = v_val / static_cast<T>(bias_correction2);

        // 4. 更新参数
        // param -= lr * m_hat / (sqrt(v_hat) + eps)
        p_val -= static_cast<T>(learning_rate) * m_hat / (sqrt(v_hat) + static_cast<T>(eps));

        // 写回参数
        params[idx] = p_val;
    }
}

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
               cudaStream_t stream) {
    
    if (size <= 0) {
        throw std::invalid_argument("adam_step: size must be > 0");
    }

    const int BLOCK_SIZE = ZEROLLM_DEFAULT_THREADS;
    const int GRID_SIZE = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // 启动 Kernel
    // 注意：我们将 bias_correction 的计算放在了 CPU 端传进来
    // 这样避免了在 GPU 每个线程中做 powf 运算
    adam_kernel<T><<<GRID_SIZE, BLOCK_SIZE, 0, stream>>>(
        params, 
        grads, 
        m, 
        v, 
        size, 
        learning_rate, 
        beta1, 
        beta2, 
        eps,
        bias_correction1,
        bias_correction2
    );

    CHECK(cudaGetLastError(), "adam_kernel launch failed");
}

// 显式实例化
template void adam_step<float>(float* params, 
                               const float* grads, 
                               float* m, 
                               float* v, 
                               int size, 
                               float learning_rate, 
                               float beta1, 
                               float beta2, 
                               float eps, 
                               float bias_correction1,
                               float bias_correction2,
                               cudaStream_t stream);