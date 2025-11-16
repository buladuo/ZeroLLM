// config.hpp
#pragma once

/**
 * @brief ZeroLLM 模型配置结构体
 * 
 * 该结构体封装了构建 ZeroLLM 模型所需的所有超参数，
 * 包括词汇表大小、嵌入维度、层数等核心配置信息。
 */
struct ZeroLLMConfig {
    int vocab_size;         // 词汇表大小
    int embed_dim;          // 嵌入维度
    int num_layers;         // Transformer 层数
    int num_heads;          // 注意力头数
    int ff_hidden_dim;      // 前馈网络隐藏层维度
    int max_seq_len;        // 最大序列长度
    bool with_grad;         // 是否启用梯度计算

    /**
     * @brief ZeroLLM 配置构造函数
     * @param vocab_size 词汇表大小
     * @param embed_dim 嵌入维度
     * @param num_layers Transformer 层数
     * @param num_heads 注意力头数
     * @param ff_hidden_dim 前馈网络隐藏层维度
     * @param max_seq_len 最大序列长度，默认为 2048
     * @param with_grad 是否启用梯度计算，默认为 false
     */
    ZeroLLMConfig(
        int vocab_size,
        int embed_dim,
        int num_layers,
        int num_heads,
        int ff_hidden_dim,
        int max_seq_len = 2048,
        bool with_grad = false
    ) : vocab_size(vocab_size),
        embed_dim(embed_dim),
        num_layers(num_layers),
        num_heads(num_heads),
        ff_hidden_dim(ff_hidden_dim),
        max_seq_len(max_seq_len),
        with_grad(with_grad) {}
};

/**
 * @brief 创建26M参数的ZeroLLM配置
 * 
 * 配置参数：
 * - vocab_size: 6400
 * - embed_dim: 512
 * - num_layers: 6
 * - num_heads: 8
 * - ff_hidden_dim: 2048
 * - max_seq_len: 2048
 * 
 * 参数量计算：
 * - Embedding: 6400 * 512 = 3.28M
 * - Transformer层 (6层):
 *   - MHA: 4 * (512^2 + 512) = 1.05M per layer
 *   - FFN: 512*2048 + 2048 + 2048*512 + 512 = 2.10M per layer
 *   - LayerNorm: 2 * 2 * 512 = 2K per layer
 *   - 每层总计: ~3.15M
 *   - 6层总计: ~18.9M
 * - Output LayerNorm: 2 * 512 = 1K
 * - 总计: ~22.2M (加上一些其他参数，约26M)
 * 
 * @return ZeroLLMConfig配置对象
 */
inline ZeroLLMConfig create_zerollm_26m_config() {
    return ZeroLLMConfig(
        6400,    // vocab_size
        512,     // embed_dim
        6,       // num_layers
        8,       // num_heads
        2048,    // ff_hidden_dim
        2048,    // max_seq_len
        true     // with_grad
    );
}