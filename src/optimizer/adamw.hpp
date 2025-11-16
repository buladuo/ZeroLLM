#pragma once
#include "adam.hpp"
#include <cuda_runtime.h>

/**
 * @brief AdamW优化器
 * 
 * 实现AdamW优化算法，将权重衰减与梯度更新解耦
 * 参考论文: Decoupled Weight Decay Regularization (https://arxiv.org/abs/1711.05101)
 */
class AdamW : public Adam {
private:
    float weight_decay_;          // 权重衰减系数

public:
    /**
     * @brief AdamW构造函数
     * @param weight_decay 权重衰减系数
     * @param beta1 一阶矩估计的指数衰减率，默认为0.9
     * @param beta2 二阶矩估计的指数衰减率，默认为0.999
     * @param eps 用于数值稳定性的小常数，默认为1e-8
     */
    explicit AdamW(float weight_decay, float beta1 = 0.9f, float beta2 = 0.999f, float eps = 1e-8f);
    
    /**
     * @brief 更新参数
     * @param param 参数指针
     * @param grad 梯度指针
     * @param num_elements 参数数量
     * @param learning_rate 学习率
     */
    void step(float* param, const float* grad, int num_elements, float learning_rate) override;
};