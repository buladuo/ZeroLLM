#pragma once
#include <cuda_runtime.h>

/**
 * @brief Optimizer基类
 * 
 * 所有优化器的基类，定义了优化器的接口
 */
class Optimizer {
public:
    /**
     * @brief 构造函数
     */
    Optimizer() = default;
    
    /**
     * @brief 虚析构函数
     */
    virtual ~Optimizer() = default;
    
    /**
     * @brief 更新参数的接口
     * @param param 参数指针
     * @param grad 梯度指针
     * @param num_elements 参数数量
     * @param learning_rate 学习率
     */
    virtual void step(float* param, const float* grad, int num_elements, float learning_rate) = 0;
    
    /**
     * @brief 清除优化器状态（如动量等）
     */
    virtual void zero_grad() {}
};


/**
 * @brief 优化器类型枚举
 */
enum class OptimizerType {
    SGD,
    Adam,
    AdamW
};

/**
 * @brief 优化器配置结构体
 * 
 * 包含优化器类型和相应参数的配置信息
 */
struct OptimizerConfig {
    OptimizerType type;
    
    // SGD参数
    float momentum;
    
    // Adam/AdamW参数
    float beta1;
    float beta2;
    float eps;
    
    // AdamW参数
    float weight_decay;
    
    /**
     * @brief 构造函数，设置默认参数值
     */
    OptimizerConfig() 
        : type(OptimizerType::SGD),
          momentum(0.0f),
          beta1(0.9f),
          beta2(0.999f),
          eps(1e-8f),
          weight_decay(0.0f) {}
    
    /**
     * @brief 创建SGD配置
     * @param momentum 动量系数
     */
    static OptimizerConfig createSGD(float momentum = 0.0f) {
        OptimizerConfig config;
        config.type = OptimizerType::SGD;
        config.momentum = momentum;
        return config;
    }
    
    /**
     * @brief 创建Adam配置
     * @param beta1 一阶矩估计的指数衰减率
     * @param beta2 二阶矩估计的指数衰减率
     * @param eps 数值稳定性小常数
     */
    static OptimizerConfig createAdam(float beta1 = 0.9f, float beta2 = 0.999f, float eps = 1e-8f) {
        OptimizerConfig config;
        config.type = OptimizerType::Adam;
        config.beta1 = beta1;
        config.beta2 = beta2;
        config.eps = eps;
        return config;
    }
    
    /**
     * @brief 创建AdamW配置
     * @param weight_decay 权重衰减系数
     * @param beta1 一阶矩估计的指数衰减率
     * @param beta2 二阶矩估计的指数衰减率
     * @param eps 数值稳定性小常数
     */
    static OptimizerConfig createAdamW(float weight_decay, float beta1 = 0.9f, float beta2 = 0.999f, float eps = 1e-8f) {
        OptimizerConfig config;
        config.type = OptimizerType::AdamW;
        config.weight_decay = weight_decay;
        config.beta1 = beta1;
        config.beta2 = beta2;
        config.eps = eps;
        return config;
    }
};