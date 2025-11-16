#pragma once
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>
#include "mha.hpp"
#include "linear.hpp"
#include "relu.hpp"
#include "layernorm.hpp"
#include "feedward.hpp"

/**
 * @brief Transformer解码器块类
 * 
 * 实现标准的Transformer解码器块结构:
 * 输入 -> 自注意力(因果) -> Add&Norm -> 前馈网络 -> Add&Norm -> 输出
 */
class TransformerDecoderBlock {
public:
    static constexpr const char* TYPE_NAME = "TransformerDecoderBlock";
private:
    int embed_dim_;                 // d_model维度
    int num_heads_;                 // 多头注意力的头数
    int ff_hidden_dim_;             // 前向传播隐藏层维度
    
    MultiHeadAttention* self_attn_; // 自注意力层（带因果掩码）
    LayerNorm* ln1_;                // 第一层归一化
    LayerNorm* ln2_;                // 第二层归一化
    
    FeedForward* ff_;               // 前馈网络
    
    float* self_attn_output_;       // 自注意力输出 [batch_size, seq_len, embed_dim]
    float* ln1_output_;             // 第一层归一化输出 [batch_size, seq_len, embed_dim]
    float* ff_output_;              // 前馈网络输出 [batch_size, seq_len, embed_dim]
    float* ln2_output_;             // 第二层归一化输出 [batch_size, seq_len, embed_dim]
    
    int last_batch_size_;           // 最后一次的batch_size
    int last_seq_len_;              // 最后一次的seq_len

public:
    /**
     * @brief 构造函数
     * @param embed_dim 嵌入维度
     * @param num_heads 头的数量
     * @param ff_hidden_dim 前向传播隐藏层的维度
     * @param with_grad 是否需要梯度
     */
    TransformerDecoderBlock(int embed_dim, int num_heads, int ff_hidden_dim, bool with_grad = false);
    ~TransformerDecoderBlock();

    /**
     * @brief 前向传播
     * @param output 输出 [batch_size, seq_len, embed_dim]
     * @param input 输入 [batch_size, seq_len, embed_dim]
     * @param batch_size batch_size
     * @param seq_len seq_len
     */
    void forward(float* output, const float* input, int batch_size, int seq_len);
    
    /**
     * @brief 反向传播
     * @param d_input 输入的梯度 [batch_size, seq_len, embed_dim]
     * @param d_output 输出的梯度 [batch_size, seq_len, embed_dim]
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
     * @brief 获取自注意力层指针
     * @return 自注意力层指针
     */
    MultiHeadAttention* self_attn() { return self_attn_; }
    
    /**
     * @brief 获取前馈网络层指针
     * @return 前馈网络层指针
     */
    FeedForward* ff() { return ff_; }
    
    /**
     * @brief 清零所有梯度
     */
    void zero_grad();
};