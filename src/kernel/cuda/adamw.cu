#include "adamw.cuh"
#include <iostream>
#include "config.hpp"

/**
 * @brief AdamW Kernel
 */
template<typename T>
__global__ void adamw_kernel(T* __restrict__ params,
                             const T* __restrict__ grads,
                             T* __restrict__ m,
                             T* __restrict__ v,
                             int size,
                             float learning_rate,
                             float weight_decay,
                             float beta1,
                             float beta2,
                             float eps,
                             float bias_correction1,
                             float bias_correction2) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        // 读取全局内存到寄存器
        T p_val = params[idx];
        T g_val = grads[idx];
        T m_val = m[idx];
        T v_val = v[idx];

        // ------------------------------------------------
        // 1. 更新矩估计 (Standard Adam Logic)
        // ------------------------------------------------
        
        // m = beta1 * m + (1 - beta1) * g
        m_val = static_cast<T>(beta1) * m_val + static_cast<T>(1.0f - beta1) * g_val;

        // v = beta2 * v + (1 - beta2) * g * g
        v_val = static_cast<T>(beta2) * v_val + static_cast<T>(1.0f - beta2) * g_val * g_val;

        // 写回 m 和 v
        m[idx] = m_val;
        v[idx] = v_val;

        // ------------------------------------------------
        // 2. 计算更新量
        // ------------------------------------------------

        // 偏差修正
        T m_hat = m_val / static_cast<T>(bias_correction1);
        T v_hat = v_val / static_cast<T>(bias_correction2);

        // 计算 Adam 的自适应步长部分
        T adam_step = static_cast<T>(learning_rate) * m_hat / (sqrt(v_hat) + static_cast<T>(eps));

        // 计算 Weight Decay 部分 (Decoupled)
        // param_decay = lr * weight_decay * param
        T decay_step = 0;
        if (weight_decay != 0.0f) {
            decay_step = static_cast<T>(learning_rate) * static_cast<T>(weight_decay) * p_val;
        }

        // ------------------------------------------------
        // 3. 应用更新
        // ------------------------------------------------
        
        // 最终参数 = 原参数 - 权重衰减项 - Adam梯度项
        // p_val -= decay_step;
        // p_val -= adam_step;
        p_val -= (decay_step + adam_step);

        // 写回参数 (只写一次 Global Memory)
        params[idx] = p_val;
    }
}

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
                cudaStream_t stream) {
    
    if (size <= 0) {
        throw std::invalid_argument("adamw_step: size must be > 0");
    }

    const int BLOCK_SIZE = ZEROLLM_DEFAULT_THREADS;
    const int GRID_SIZE = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;

    adamw_kernel<T><<<GRID_SIZE, BLOCK_SIZE, 0, stream>>>(
        params, 
        grads, 
        m, 
        v, 
        size, 
        learning_rate, 
        weight_decay,
        beta1, 
        beta2, 
        eps,
        bias_correction1,
        bias_correction2
    );

    CHECK(cudaGetLastError(), "adamw_kernel launch failed");
}

// 显式实例化
template void adamw_step<float>(float* params, 
                                const float* grads, 
                                float* m, 
                                float* v, 
                                int size, 
                                float learning_rate, 
                                float weight_decay,
                                float beta1, 
                                float beta2, 
                                float eps, 
                                float bias_correction1,
                                float bias_correction2,
                                cudaStream_t stream);