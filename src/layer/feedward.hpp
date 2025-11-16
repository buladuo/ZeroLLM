#pragma once
#include <cuda_runtime.h>
#include "linear.hpp"
#include "relu.hpp"
#include "optimizer.hpp"
/**
 * @brief 前馈神经网络层类
 * 
 * 实现标准的前馈神经网络结构:
 * 输入 -> Linear1 -> ReLU -> Linear2 -> 输出
 */
class FeedForward {
public:
    static constexpr const char* TYPE_NAME = "FeedForward";
private:
    int embed_dim_;         // 输入嵌入维度
    int ff_hidden_dim_;     // 前馈网络隐藏层维度
    
    Linear* ff1_;           // 第一个线性层
    ReLU* relu_;            // ReLU激活函数
    Linear* ff2_;           // 第二个线性层
    
    float* ff1_output_;     // 第一个线性层输出缓冲区
    float* relu_output_;    // ReLU输出缓冲区
    float* ff2_output_;     // 第二个线性层输出缓冲区
    
    int last_batch_size_;   // 上一次前向传播的批次大小
    int last_seq_len_;      // 上一次前向传播的序列长度

public:
    /**
     * @brief 前馈网络构造函数
     * @param embed_dim 输入嵌入维度
     * @param ff_hidden_dim 前馈网络隐藏层维度
     * @param with_grad 是否需要梯度计算
     */
    FeedForward(int embed_dim, int ff_hidden_dim, bool with_grad = false);
    
    /**
     * @brief 前馈网络析构函数
     */
    ~FeedForward();
    
    /**
     * @brief 前向传播
     * @param output 输出数据 [batch_size, seq_len, embed_dim]
     * @param input 输入数据 [batch_size, seq_len, embed_dim]
     * @param batch_size 批次大小
     * @param seq_len 序列长度
     */
    void forward(float* output, const float* input, int batch_size, int seq_len);
    
    /**
     * @brief 反向传播
     * @param d_input 输入梯度 [batch_size, seq_len, embed_dim]
     * @param d_output 输出梯度 [batch_size, seq_len, embed_dim]
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
     * @brief 获取第一个线性层指针
     * @return 第一个线性层指针
     */
    Linear* ff1() { return ff1_; }
    
    /**
     * @brief 获取第二个线性层指针
     * @return 第二个线性层指针
     */
    Linear* ff2() { return ff2_; }
    
    /**
     * @brief 清零所有梯度
     */
    void zero_grad();
};