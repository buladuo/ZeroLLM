#pragma once
#include <cuda_runtime.h>
#include <vector>
#include <stdexcept>
#include <string>
#include "zerollm_config.hpp"
#include "embedding.hpp"
#include "transformer_decoder.hpp"
#include "layernorm.hpp"
#include "matmul.cuh"
#include "transpose.cuh"
#include "add.cuh"
#include "config.hpp"
#include "module.hpp"
/**
 * @brief ZeroLLM 语言模型类
 * 
 * 实现一个完整的语言模型，包含：
 * - Embedding层：词嵌入 + 位置编码
 * - Transformer Decoder：多层Transformer解码器
 * - Output层：与Embedding共享权重的输出层（转置）
 */
class ZeroLLM : public Module {
public:
    static constexpr const char* TYPE_NAME = "ZeroLLM";
private:
    ZeroLLMConfig config_;                    // 模型配置
    
    Embedding* embedding_;                     // 嵌入层
    TransformerDecoder* decoder_;              // Transformer解码器
    LayerNorm* output_ln_;                    // 输出层归一化（可选）
    
    // 临时缓冲区
    float* hidden_states_;                    // 隐藏状态 [batch_size, seq_len, embed_dim]
    float* logits_;                           // 输出logits [batch_size, seq_len, vocab_size]
    
    int last_batch_size_;                     // 最后一次的batch_size
    int last_seq_len_;                        // 最后一次的seq_len
    
    /**
     * @brief 计算模型参数量
     * @return 参数量（以百万为单位）
     */
    double calculate_num_params() const;

public:
    /**
     * @brief ZeroLLM 构造函数
     * @param config 模型配置
     */
    ZeroLLM(const ZeroLLMConfig& config);
    
    
    /**
     * @brief ZeroLLM 析构函数
     */
    ~ZeroLLM();
    
    /**
     * @brief 前向传播
     * @param logits 输出logits [batch_size, seq_len, vocab_size]
     * @param input_ids 输入token IDs [batch_size, seq_len]
     * @param batch_size 批次大小
     * @param seq_len 序列长度
     */
    void forward(float* logits, const int* input_ids, int batch_size, int seq_len);
    
    /**
     * @brief 反向传播
     * @param input_ids 输入token IDs [batch_size, seq_len]
     * @param d_logits 输出logits的梯度 [batch_size, seq_len, vocab_size]
     * @param batch_size 批次大小
     * @param seq_len 序列长度
     */
    void backward(const int* input_ids, const float* d_logits, int batch_size, int seq_len);

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
     * @brief 清零所有梯度
     */
    void zero_grad();
    
    /**
     * @brief 获取嵌入层指针
     * @return 嵌入层指针
     */
    Embedding* embedding() { return embedding_; }
    
    /**
     * @brief 获取解码器指针
     * @return 解码器指针
     */
    TransformerDecoder* decoder() { return decoder_; }
    
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
     * @brief 获取模型配置
     * @return 模型配置的常量引用
     */
    const ZeroLLMConfig& config() const { return config_; }
    
    /**
     * @brief 获取模型参数量（以百万为单位）
     * @return 参数量（M）
     */
    double num_params() const { return calculate_num_params(); }

    std::string type_name() const override { return TYPE_NAME; }
};

