#pragma once
#include <cuda_runtime.h>
#include <vector>
#include <stdexcept>
#include <string>
#include "transformer_block.hpp"
#include "config.hpp"

/**
 * @brief Transformer解码器类
 * 
 * 实现Transformer解码器，用于堆叠多个TransformerDecoderBlock层
 */
class TransformerDecoder {
public:
    static constexpr const char* TYPE_NAME = "TransformerDecoder";
private:
    int num_layers_;                        // 解码器层数
    int embed_dim_;                         // 嵌入维度
    int num_heads_;                         // 多头注意力的头数
    int ff_hidden_dim_;                     // 前向传播隐藏层维度
    
    std::vector<TransformerDecoderBlock*> layers_;  // 解码器层列表
    float* layers_output_;                  // 层间输出缓冲区 [batch_size, seq_len, embed_dim]
    
    int last_batch_size_;                   // 最后一次的batch_size
    int last_seq_len_;                      // 最后一次的seq_len

public:
    /**
     * @brief 构造函数
     * @param num_layers 解码器层数
     * @param embed_dim 嵌入维度
     * @param num_heads 头的数量
     * @param ff_hidden_dim 前向传播隐藏层的维度
     * @param with_grad 是否需要梯度
     */
    TransformerDecoder(int num_layers, int embed_dim, int num_heads, int ff_hidden_dim, bool with_grad = false);
    ~TransformerDecoder();

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
     * @brief 获取指定层的指针
     * @param layer_index 层索引
     * @return 指定层的指针
     */
    TransformerDecoderBlock* layer(int layer_index) {
        if (layer_index < 0 || layer_index >= num_layers_) {
            throw std::out_of_range("Layer index out of range");
        }
        return layers_[layer_index];
    }
    
    /**
     * @brief 获取解码器层数
     * @return 解码器层数
     */
    int num_layers() const { return num_layers_; }
    
    /**
     * @brief 获取嵌入维度
     * @return 嵌入维度
     */
    int embed_dim() const { return embed_dim_; }
    
    /**
     * @brief 清零所有梯度
     */
    void zero_grad();
};