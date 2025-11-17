// layernorm.hpp
#pragma once
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

#include "optimizer.hpp"
#include "module.hpp"
/**
 * @brief Layer Normalization 层类
 * 
 * 实现层归一化操作，对每个样本的特征维度进行归一化处理
 * 公式: y = gamma * (x - mean) / sqrt(var + eps) + beta
 */
class LayerNorm : public Module {
public:
    static constexpr const char* TYPE_NAME = "LayerNorm";
private:
    float* gamma_;          // 可学习的缩放参数 [feature_size]
    float* beta_;           // 可学习的偏移参数 [feature_size]
    int feature_size_;      // 特征维度大小
    bool with_grad_;        // 是否需要计算梯度
    
    // 梯度
    float* d_gamma_;        // gamma参数的梯度 [feature_size]
    float* d_beta_;         // beta参数的梯度 [feature_size]
    
    // 保存前向传播时的中间结果用于反向传播
    float* mean_;           // 每个样本的均值 [batch_size]
    float* rstd_;           // 每个样本的 1 / sqrt(variance + eps) [batch_size]
    
    const float* input_;    // 前向传播时的输入数据指针
    int last_batch_size_;   // 上一次前向传播的批次大小
    
    float eps_;             // 防止除零的小常数

    Optimizer* gamma_optimizer_;
    Optimizer* beta_optimizer_;

public:
    /**
     * @brief LayerNorm构造函数
     * @param feature_size 特征维度大小
     * @param with_grad 是否需要梯度计算
     * @param eps 防止除零的小常数
     */
    LayerNorm(int feature_size, bool with_grad = false, float eps = 1e-5);
    ~LayerNorm();

    /**
     * @brief 清零梯度
     */
    void zero_grad();
    
    /**
     * @brief 前向传播
     * @param output 输出数据 [batch_size, feature_size]
     * @param input 输入数据 [batch_size, feature_size]
     * @param batch_size 批次大小
     */
    void forward(float* output, const float* input, int batch_size);
    
    /**
     * @brief 反向传播
     * @param d_input 输入梯度 [batch_size, feature_size]
     * @param d_output 输出梯度 [batch_size, feature_size]
     */
    void backward(float* d_input, const float* d_output);

    /**
     * @brief 保存模型参数
     * 
     * @param path 保存路径
     */
    void save(const std::string& path);
    
    /**
     * @brief 加载模型参数
     * 
     * @param path 加载路径
     */
    void load(const std::string& path);

    /**
     * @brief 设置优化器
     * @param config 优化器配置
     */
    void set_optimizer(OptimizerConfig config);

    /**
     * @brief 优化器步进
     * @param learning_rate 学习率
     */
    void step(float learning_rate);
    
    /**
     * @brief 获取gamma参数指针
     * @return gamma参数指针
     */
    float* gamma();
    
    /**
     * @brief 获取beta参数指针
     * @return beta参数指针
     */
    float* beta();
    
    /**
     * @brief 获取gamma梯度指针
     * @return gamma梯度指针
     */
    float* d_gamma();
    
    /**
     * @brief 获取beta梯度指针
     * @return beta梯度指针
     */
    float* d_beta();
    
    /**
     * @brief 获取特征维度大小
     * @return 特征维度大小
     */
    int feature_size() const { return feature_size_; }

    std::string type_name() const override { return TYPE_NAME; }
};