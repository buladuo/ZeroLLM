#pragma once
#include "optimizer.hpp"
#include <cuda_runtime.h>

/**
 * @brief 随机梯度下降优化器
 * 
 * 实现标准的SGD优化算法，支持动量项
 */
class SGD : public Optimizer {
private:
    float momentum_;              // 动量系数
    float* velocity_;             // 速度缓冲区（用于动量）
    bool use_momentum_;           // 是否使用动量
    int last_num_elements_;       // 上次更新的参数数量

public:
    /**
     * @brief SGD构造函数
     * @param momentum 动量系数，默认为0（不使用动量）
     */
    explicit SGD(float momentum = 0.0f);
    
    /**
     * @brief SGD析构函数
     */
    ~SGD();
    
    /**
     * @brief 更新参数
     * @param param 参数指针
     * @param grad 梯度指针
     * @param num_elements 参数数量
     * @param learning_rate 学习率
     */
    void step(float* param, const float* grad, int num_elements, float learning_rate) override;
    
    /**
     * @brief 清零动量缓冲区
     */
    void zero_grad() override;
};