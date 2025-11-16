#pragma once
#include <cuda_runtime.h>
#include "activation.hpp"

/**
 * @brief ReLU激活函数类
 * 
 * 实现ReLU激活函数: f(x) = max(0, x)
 * 支持前向传播和反向传播
 */
class ReLU : public Activation {
public:
    static constexpr const char* TYPE_NAME = "Activation";
public:
    /**
     * @brief ReLU构造函数
     */
    ReLU();
    
    /**
     * @brief ReLU析构函数
     */
    ~ReLU();

    /**
     * @brief 前向传播 RELU 
     * 
     * @param output 输出数据存放指针 [batch_size, features]
     * @param input 输入数据存放指针 [batch_size, features]
     * @param batch_size 批次大小
     * @param features 特征数量
     */
    void forward(float* output, const float* input, int batch_size, int features) override;
    
    /**
     * @brief 反向传播
     * 
     * @param d_input 输入梯度 [batch_size, features]
     * @param d_output 输出梯度 [batch_size, features]
     */
    void backward(float* d_input, const float* d_output) override;
    
    /**
     * @brief 清零梯度
     * 
     * ReLU层本身没有可学习参数，因此不需要清零梯度
     */
    void zero_grad();
};