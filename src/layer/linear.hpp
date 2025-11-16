#pragma once
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>
#include "optimizer.hpp"

/**
 * @brief 线性层(全连接层)类
 * 
 * 实现线性变换: Y = X * W^T + b
 * 支持前向传播和反向传播，可选择是否使用偏置项
 */
class Linear {
public:
    static constexpr const char* TYPE_NAME = "Linear";
private:
    float* weight_;             // 权重矩阵 [out_features, in_features]
    float* bias_;               // 偏置向量 [out_features]
    bool use_bias_;             // 是否使用偏置
    bool with_grad_;            // 是否需要计算梯度
    int in_features_;           // 输入特征数
    int out_features_;          // 输出特征数

    float* d_weight_;           // 权重梯度 [out_features, in_features]
    float* d_bias_;             // 偏置梯度 [out_features]
    const float* input_;        // 前向传播时的输入数据指针
    int last_batch_size_;       // 上一次前向传播的批次大小

    Optimizer* weight_optimizer_;      // 优化器
    Optimizer* bias_optimizer_;

    /**
     * @brief 初始化权重
     * 
     * 使用Xavier初始化方法初始化权重
     */
    void initializeWeights();

public:
    /**
     * @brief 线性层构造函数
     * @param in_features 输入特征数
     * @param out_features 输出特征数
     * @param use_bias 是否使用偏置项
     * @param with_grad 是否需要梯度计算
     */
    Linear(int in_features, int out_features, bool use_bias = true, bool with_grad = false);
    ~Linear();

    /**
     * @brief 清零梯度
     */
    void zero_grad();
    
    /**
     * @brief 前向传播
     * @param output 输出数据 [batch_size, out_features]
     * @param input 输入数据 [batch_size, in_features]
     * @param batch_size 批次大小
     */
    void forward(float* output, const float* input, int batch_size);
    
    /**
     * @brief 反向传播
     * @param d_input 输入梯度 [batch_size, in_features]
     * @param d_output 输出梯度 [batch_size, out_features]
     */
    void backward(float* d_input, const float* d_output);

    /**
     * @brief 设置优化器
     * @param optimizer 优化器
     */
    void set_optimizer(OptimizerConfig config);

    /**
     * @brief 优化权重
     * @param learning_rate 学习率
     */
    void step(float learning_rate);

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
     * @brief 获取权重指针
     * @return 权重指针
     */
    float* weight();
    
    /**
     * @brief 获取偏置指针
     * @return 偏置指针
     */
    float* bias();
    
    /**
     * @brief 获取权重梯度指针
     * @return 权重梯度指针
     */
    float* d_weight();
    
    /**
     * @brief 获取偏置梯度指针
     * @return 偏置梯度指针
     */
    float* d_bias();
};