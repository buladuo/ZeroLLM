
#include "sgd.cuh"
#include <iostream>
#include "config.hpp" 

/**
 * @brief SGD Kernel
 * * 处理三种情况：
 * 1. 仅 SGD: param -= lr * grad
 * 2. SGD + Weight Decay
 * 3. SGD + Momentum (+ Weight Decay)
 */
template<typename T>
__global__ void sgd_kernel(T* __restrict__ params,
                           const T* __restrict__ grads,
                           T* __restrict__ velocity,
                           int size,
                           float learning_rate,
                           float momentum,
                           float weight_decay) {
    // 全局索引计算
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        // 读取当前参数和梯度
        T p = params[idx];
        T g = grads[idx];

        // 1. 应用 Weight Decay (L2 Regularization)
        // g = g + weight_decay * p
        if (weight_decay != 0.0f) {
            g += static_cast<T>(weight_decay) * p;
        }

        // 2. 应用 Momentum
        if (momentum != 0.0f && velocity != nullptr) {
            // 读取上一时刻的速度
            T v = velocity[idx];
            
            // v_t = momentum * v_{t-1} + g_t
            v = static_cast<T>(momentum) * v + g;
            
            // 写回速度缓存
            velocity[idx] = v;
            
            // 在有动量的情况下，更新方向就是 v
            g = v;
        }

        // 3. 更新参数
        // p_t = p_{t-1} - lr * update_direction
        p -= static_cast<T>(learning_rate) * g;

        // 写回参数
        params[idx] = p;
    }
}

template<typename T>
void sgd_step(T* params, 
              const T* grads, 
              T* velocity, 
              int size, 
              float learning_rate, 
              float momentum, 
              float weight_decay, 
              cudaStream_t stream) {
    
    if (size <= 0) {
        throw std::invalid_argument("sgd_step: size must be > 0");
    }

    const int BLOCK_SIZE = ZEROLLM_DEFAULT_THREADS;
    const int GRID_SIZE = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // 启动 Kernel
    sgd_kernel<T><<<GRID_SIZE, BLOCK_SIZE, 0, stream>>>(
        params, 
        grads, 
        velocity, 
        size, 
        learning_rate, 
        momentum, 
        weight_decay
    );

    CHECK(cudaGetLastError(), "sgd_kernel launch failed");
}

// 显式实例化 (根据需要添加 double)
template void sgd_step<float>(float* params, 
                              const float* grads, 
                              float* velocity, 
                              int size, 
                              float learning_rate, 
                              float momentum, 
                              float weight_decay, 
                              cudaStream_t stream);