#pragma once
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>
#include "linear.hpp"

/**
 * @brief 多头注意力机制类
 * 
 * 实现多头自注意力机制，支持因果注意力(用于解码器)
 * 计算流程:
 * 1. 通过线性投影得到Q, K, V
 * 2. 计算注意力分数: QK^T / sqrt(d_k)
 * 3. 应用掩码(可选)
 * 4. 应用softmax
 * 5. 计算输出: softmax(QK^T/sqrt(d_k)) * V
 * 6. 通过输出投影层
 */
class MultiHeadAttention {
public:
    static constexpr const char* TYPE_NAME = "MultiHeadAttention";
private:
    int embed_dim_;             // 输入 embedding 的维度
    int num_heads_;             // 多头注意力的头数
    int head_dim_;              // 每个头的维度 (embed_dim / num_heads)
    bool is_causal_;            // 是否为因果注意力（用于解码器）
    
    Linear* q_proj_;            // 查询投影层
    Linear* k_proj_;            // 键投影层
    Linear* v_proj_;            // 值投影层
    Linear* out_proj_;          // 输出投影层
    
    float* Q_;                  // 查询投影结果 [batch_size, seq_len, embed_dim]
    float* K_;                  // 键投影结果 [batch_size, seq_len, embed_dim]
    float* V_;                  // 值投影结果 [batch_size, seq_len, embed_dim]
    float* attn_output_;        // 注意力输出 [batch_size, seq_len, embed_dim]
    float* attention_scores_;   // 注意力分数 [batch_size, num_heads, seq_len, seq_len]，用于反向传播
    bool* mask_;                // 注意力掩码 [seq_len, seq_len]
    
    int last_batch_size_;       // 最后一次的 batch_size
    int last_seq_len_;          // 最后一次的 seq_len

public:
    /**
     * @brief 多头注意力构造函数
     * @param embed_dim 输入 embedding 的维度
     * @param num_heads 多头注意力的头数
     * @param with_grad 是否需要梯度计算
     * @param is_causal 是否为因果注意力（用于解码器）
     */
    MultiHeadAttention(int embed_dim, int num_heads, bool with_grad = false, bool is_causal = false);
    ~MultiHeadAttention();

    /**
     * @brief 前向传播
     * @param output 输出结果 [batch_size, seq_len, embed_dim]
     * @param input 输入 [batch_size, seq_len, embed_dim]
     * @param batch_size batch_size
     * @param seq_len seq_len
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
     * @brief 设置mask
     * @param mask mask [seq_len, seq_len]
     */
    void set_mask(const bool* mask);
    
    /**
     * @brief 获取嵌入维度
     * @return 嵌入维度
     */
    int embed_dim() const { return embed_dim_; }
    
    /**
     * @brief 获取头数
     * @return 头数
     */
    int num_heads() const { return num_heads_; }
    
    /**
     * @brief 获取每个头的维度
     * @return 每个头的维度
     */
    int head_dim() const { return head_dim_; }
    
    /**
     * @brief 是否为因果注意力
     * @return 是否为因果注意力
     */
    bool is_causal() const { return is_causal_; }
    
    /**
     * @brief 获取查询投影层
     * @return 查询投影层指针
     */
    Linear* q_proj() { return q_proj_; }
    
    /**
     * @brief 获取键投影层
     * @return 键投影层指针
     */
    Linear* k_proj() { return k_proj_; }
    
    /**
     * @brief 获取值投影层
     * @return 值投影层指针
     */
    Linear* v_proj() { return v_proj_; }
    
    /**
     * @brief 获取输出投影层
     * @return 输出投影层指针
     */
    Linear* out_proj() { return out_proj_; }
    
    /**
     * @brief 清零所有梯度
     */
    void zero_grad();
};