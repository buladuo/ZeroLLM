// embedding.hpp
#pragma once
#include <cuda_runtime.h>
#include <vector>
#include "optimizer.hpp"
#include <string>
#include "module.hpp"

/**
 * @brief 嵌入层类，用于将token IDs转换为嵌入向量表示
 * 
 * 该类实现了词嵌入和位置编码的组合，将离散的token IDs映射为连续的向量表示。
 * 使用CUDA加速计算，支持前向传播和反向传播。
 */
class Embedding : public Module {
public:
    static constexpr const char* TYPE_NAME = "Embedding";

private:
    int vocab_size_;                // 词汇表大小
    int embed_dim_;                 // 嵌入维度
    float* embedding_table_;        // 词汇嵌入表 [vocab_size, embed_dim]
    float* pos_encoding_table_;     // 位置编码表 [max_seq_len, embed_dim]
    int max_seq_len_;               // 最大序列长度
    
    // GPU上的数据
    float* d_embedding_table_;      // GPU上的嵌入表
    float* d_pos_encoding_table_;   // GPU上的位置编码表
    
    // 梯度相关
    float* d_embedding_table_grad_; // GPU上的嵌入表梯度
    bool with_grad_;                // 是否需要计算梯度
    
    Optimizer* optimizer_;          // 优化器

    /**
     * @brief 初始化嵌入表
     * 
     * 使用随机值初始化词汇嵌入表
     */
    void initializeEmbeddingTable();
    
    /**
     * @brief 初始化位置编码表
     * 
     * 使用正弦位置编码初始化位置编码表
     */
    void initializePositionalEncodingTable();
    
public:
    /**
     * @brief 嵌入层构造函数
     * @param vocab_size 词表大小
     * @param embed_dim 嵌入维度
     * @param max_seq_len 最大序列长度
     * @param with_grad 是否需要梯度
     */
    Embedding(int vocab_size, int embed_dim, int max_seq_len = 2048, bool with_grad = false);
    ~Embedding();
    
    /**
     * @brief 前向传播，将token IDs转换为嵌入表示
     * 
     * @param output 输出嵌入 [batch_size, seq_len, embed_dim]
     * @param input 输入token IDs [batch_size, seq_len]
     * @param batch_size 批处理大小
     * @param seq_len 序列长度
     */
    void forward(float* output, const int* input, int batch_size, int seq_len);
    
    /**
     * @brief 反向传播，计算梯度
     * 
     * @param input 输入token IDs [batch_size, seq_len]
     * @param d_output 输出梯度 [batch_size, seq_len, embed_dim]
     * @param batch_size 批处理大小
     * @param seq_len 序列长度
     */
    void backward(const int* input, const float* d_output, int batch_size, int seq_len, bool accumulate = false);

    /**
     * @brief 初始化优化器
     * 
     * @param optimizer 优化器
     */
    void set_optimizer(OptimizerConfig config);

    /**
     * @brief 优化参数
     * 
     * @param learning_rate 学习率
     */
    void step(float learning_rate);
    
    /**
     * @brief 清零梯度
     */
    void zero_grad();
    
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
     * @brief 获取嵌入表梯度
     */
    float* d_embedding_table();

    /**
     * @brief 获取GPU上的嵌入表指针
     */
    float* embedding_table_device();
    const float* embedding_table_device() const;
    
    /**
     * @brief 获取嵌入维度
     */
    int embedDim() const { return embed_dim_; }
    
    /**
     * @brief 获取词汇表大小
     */
    int vocabSize() const { return vocab_size_; }

    std::string type_name() const override { return TYPE_NAME; }
};