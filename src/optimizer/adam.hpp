#pragma once
#include "optimizer.hpp"
#include <cuda_runtime.h>
#include <cstdint>
/**
 * @brief Adam优化器
 * 
 * 实现Adam优化算法
 * 参考论文: Adam: A Method for Stochastic Optimization (https://arxiv.org/abs/1412.6980)
 */
class Adam : public Optimizer {
protected:
    float beta1_;                 // 一阶矩估计的指数衰减率
    float beta2_;                 // 二阶矩估计的指数衰减率
    float eps_;                   // 用于数值稳定性的小常数
    float* m_;                    // 一阶矩估计缓冲区
    float* v_;                    // 二阶矩估计缓冲区
    int64_t t_;                   // 时间步
    int last_num_elements_;       // 上次更新的参数数量

public:
    /**
     * @brief Adam构造函数
     * @param beta1 一阶矩估计的指数衰减率，默认为0.9
     * @param beta2 二阶矩估计的指数衰减率，默认为0.999
     * @param eps 用于数值稳定性的小常数，默认为1e-8
     */
    explicit Adam(float beta1 = 0.9f, float beta2 = 0.999f, float eps = 1e-8f);
    
    /**
     * @brief Adam析构函数
     */
    ~Adam();
    
    /**
     * @brief 更新参数
     * @param param 参数指针
     * @param grad 梯度指针
     * @param num_elements 参数数量
     * @param learning_rate 学习率
     */
    void step(float* param, const float* grad, int num_elements, float learning_rate) override;
    
    /**
     * @brief 清零优化器状态
     */
    void zero_grad() override;
    
    /**
     * @brief 重置时间步
     */
    void reset_time() { t_ = 0; }
};