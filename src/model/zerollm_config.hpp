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


/**
 * @brief 58M 参数配置 (Deep Nano)
 * * 相比 26M 版本，保持宽度不变，但将深度翻倍，并略微增加 FFN 宽度。
 * 适合在显存非常紧张时尝试增加模型非线性能力。
 * * 估算: ~58M
 * 建议 Batch Size: 48 或 32
 */
inline ZeroLLMConfig create_zerollm_58m_config() {
    return ZeroLLMConfig(
        6400,    // vocab_size
        512,     // embed_dim
        12,      // num_layers (层数翻倍: 6 -> 12)
        8,       // num_heads
        2048,    // ff_hidden_dim
        2048,    // max_seq_len
        true     // with_grad
    );
}

/**
 * @brief 110M 参数配置 (Standard Small - 类似 GPT-2 Small / BERT Base)
 * * 这是验证深度学习框架最经典的"甜点"配置。
 * 能够学习到较复杂的语法和逻辑，Loss 通常能收敛到 3.0-4.0 以下。
 * * 估算: ~110M
 * - Embed: 768
 * - Layers: 12
 * - Heads: 12
 * * !注意!: 以前占用 23GB (Batch 64)，使用此模型必须将 Batch Size 降至 16 或 24。
 */
inline ZeroLLMConfig create_zerollm_110m_config() {
    return ZeroLLMConfig(
        6400,    // vocab_size
        768,     // embed_dim (主流基准宽度)
        12,      // num_layers
        12,      // num_heads (768 / 64 = 12)
        3072,    // ff_hidden_dim (通常是 embed_dim * 4)
        2048,    // max_seq_len
        true     // with_grad
    );
}

/**
 * @brief 340M 参数配置 (Medium - 类似 GPT-2 Medium)
 * * 具有较强的拟合能力，如果代码实现正确，Loss 应能轻松低于 3.0。
 * * 估算: ~340M
 * - Embed: 1024
 * - Layers: 24
 * * !警告!: 显存压力巨大。
 * 必须将 Batch Size 降至 8 甚至 4 才能在单卡 24G 上运行（取决于你的框架优化程度）。
 */
inline ZeroLLMConfig create_zerollm_340m_config() {
    return ZeroLLMConfig(
        6400,    // vocab_size
        1024,    // embed_dim
        24,      // num_layers (深度很深)
        16,      // num_heads (1024 / 64 = 16)
        4096,    // ff_hidden_dim
        2048,    // max_seq_len
        true     // with_grad
    );
}