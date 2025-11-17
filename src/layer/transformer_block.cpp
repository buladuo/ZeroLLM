#include "transformer_block.hpp"
#include <stdexcept>
#include <string>
#include "config.hpp"  // 引入后端配置头文件
#include "add.cuh"
#include "async_logger.hpp"
#include "serializer.hpp"

/**
 * @brief Transformer解码器块构造函数
 * @param embed_dim 嵌入维度
 * @param num_heads 头的数量
 * @param ff_hidden_dim 前向传播隐藏层的维度
 * @param with_grad 是否需要梯度计算
 */
TransformerDecoderBlock::TransformerDecoderBlock(int embed_dim, int num_heads, int ff_hidden_dim, bool with_grad)
    : embed_dim_(embed_dim),
      num_heads_(num_heads),
      ff_hidden_dim_(ff_hidden_dim),
      last_batch_size_(0),
      last_seq_len_(0),
      self_attn_output_(nullptr),
      ln1_output_(nullptr),
      ff_output_(nullptr),
      ln2_output_(nullptr) {
    
    LOG_DEBUG("Initializing TransformerDecoderBlock: embed_dim=" << embed_dim << ", num_heads=" << num_heads 
              << ", ff_hidden_dim=" << ff_hidden_dim);
    
    self_attn_ = new MultiHeadAttention(embed_dim, num_heads, with_grad, true);  // 使用因果注意力
    ln1_ = new LayerNorm(embed_dim, with_grad);
    ln2_ = new LayerNorm(embed_dim, with_grad);
    ff_ = new FeedForward(embed_dim, ff_hidden_dim, with_grad);
    
    register_module("self_attn", self_attn_);
    register_module("ln1", ln1_);
    register_module("ln2", ln2_);
    register_module("ff", ff_);
    
    LOG_DEBUG("TransformerDecoderBlock initialized");
}

/**
 * @brief Transformer解码器块析构函数，释放内存
 */
TransformerDecoderBlock::~TransformerDecoderBlock() {
    delete self_attn_;
    delete ln1_;
    delete ln2_;
    delete ff_;
    
    zerollm_backend::free(self_attn_output_);
    zerollm_backend::free(ln1_output_);
    zerollm_backend::free(ff_output_);
    zerollm_backend::free(ln2_output_);
}

/**
 * @brief 前向传播
 * 
 * 执行Transformer解码器块的前向计算:
 * 输入 -> 自注意力(因果) -> Add&Norm -> 前馈网络 -> Add&Norm -> 输出
 * @param output 输出 [batch_size, seq_len, embed_dim]
 * @param input 输入 [batch_size, seq_len, embed_dim]
 * @param batch_size batch_size
 * @param seq_len seq_len
 */
void TransformerDecoderBlock::forward(float* output, const float* input, int batch_size, int seq_len) {
    LOG_DEBUG("TransformerDecoderBlock forward: batch_size=" << batch_size << ", seq_len=" << seq_len);
    
    // 创建缓冲区
    if (last_batch_size_ != batch_size || last_seq_len_ != seq_len) {
        LOG_DEBUG("Resizing buffers for new batch/seq size");
        zerollm_backend::free(self_attn_output_);
        zerollm_backend::free(ln1_output_);
        zerollm_backend::free(ff_output_);
        zerollm_backend::free(ln2_output_);
        
        int total_elements = batch_size * seq_len * embed_dim_;
        
        self_attn_output_ = (float *)zerollm_backend::malloc(total_elements * sizeof(float));
        ln1_output_ = (float *)zerollm_backend::malloc(total_elements * sizeof(float));
        ff_output_ = (float *)zerollm_backend::malloc(total_elements * sizeof(float));
        ln2_output_ = (float *)zerollm_backend::malloc(total_elements * sizeof(float));
        
        last_batch_size_ = batch_size;
        last_seq_len_ = seq_len;
    }
    
    LOG_DEBUG("Computing self-attention");
    self_attn_->forward(self_attn_output_, input, batch_size, seq_len);
    
    // Add & Norm (使用后端加法)
    int total_elements = batch_size * seq_len * embed_dim_;
    LOG_DEBUG("Add & Norm after self-attention");
    add<float>(input, self_attn_output_, ln1_output_, total_elements, 0); 
    ln1_->forward(ln1_output_, ln1_output_, batch_size * seq_len);
    
    // Feed-forward
    LOG_DEBUG("Computing feed-forward");
    ff_->forward(ff_output_, ln1_output_, batch_size, seq_len);
    
    // Add & Norm (使用后端加法)
    LOG_DEBUG("Add & Norm after feed-forward");
    add<float>(ln1_output_, ff_output_, ln2_output_, total_elements, 0);
    ln2_->forward(output, ln2_output_, batch_size * seq_len);
    
    zerollm_backend::device_synchronize();
    zerollm_backend::check_last_error("TransformerDecoderBlock::forward() failed");
    LOG_DEBUG("TransformerDecoderBlock forward completed");
}



/**
 * @brief 反向传播
 *
 * 计算Transformer解码器块的梯度
 * @param d_input 输入的梯度 [batch_size, seq_len, embed_dim]
 * @param d_output 输出的梯度 [batch_size, seq_len, embed_dim]
 * @note 梯度推导过程:
 * 1. 梯度计算:
 */
void TransformerDecoderBlock::backward(float* d_input, const float* d_output) {
    LOG_DEBUG("TransformerDecoderBlock backward started");
    
    if (last_batch_size_ == 0 || last_seq_len_ == 0) {
        throw std::runtime_error("Must call forward() before backward()");
    }

    int batch_size = last_batch_size_;
    int seq_len = last_seq_len_;
    int total_elements = batch_size * seq_len * embed_dim_;

    float* d_ln2_out = nullptr;
    float* d_ff_out = nullptr;
    float* d_ff_in = nullptr; // 存储来自 Res2 残差分支的梯度
    float* d_ln1_out = nullptr;
    float* d_attn_out = nullptr;
    
    // --- 新增的临时缓冲区 ---
    float* d_ff_in_from_ffn = nullptr; // 存储来自 FFN 分支的梯度

    d_ln2_out = (float *)zerollm_backend::malloc(total_elements * sizeof(float));
    d_ff_out = (float *)zerollm_backend::malloc(total_elements * sizeof(float));
    d_ff_in = (float *)zerollm_backend::malloc(total_elements * sizeof(float));
    d_ln1_out = (float *)zerollm_backend::malloc(total_elements * sizeof(float));
    d_attn_out = (float *)zerollm_backend::malloc(total_elements * sizeof(float));
    
    // --- 新增 ---
    d_ff_in_from_ffn = (float *)zerollm_backend::malloc(total_elements * sizeof(float));

    LOG_DEBUG("Backward through LayerNorm 2");
    ln2_->backward(d_ln2_out, d_output);

    // Backward through second residual connection (分发梯度)
    // 1. FFN 分支的梯度 (作为 ff_->backward 的输入)
    zerollm_backend::memcpy(d_ff_out, d_ln2_out, total_elements * sizeof(float), zerollm_backend::CopyKind::D2D);
    // 2. 残差分支的梯度 (dL/d(LN1)_res)，先存在 d_ff_in
    zerollm_backend::memcpy(d_ff_in, d_ln2_out, total_elements * sizeof(float), zerollm_backend::CopyKind::D2D);

    // --- 修改部分开始 ---

    // Backward through feed-forward network
    // 计算 dL/d(LN1)_ffn 并将其存储在 *新的* 缓冲区中
    LOG_DEBUG("Backward through feed-forward network");
    ff_->backward(d_ff_in_from_ffn, d_ff_out);

    // 手动累加两个分支的梯度
    // d_ff_in (总) = d_ff_in (残差) + d_ff_in_from_ffn (来自FFN)
    LOG_DEBUG("Accumulating gradients from residual and FFN branches");
    add_inplace<float>(d_ff_in, d_ff_in_from_ffn, total_elements, 0);

    // Backward through first layer norm
    // 现在的 d_ff_in 包含正确的总梯度
    LOG_DEBUG("Backward through LayerNorm 1");
    ln1_->backward(d_ln1_out, d_ff_in);

    // Backward through first residual connection (分发梯度)
    // 复制梯度到两个分支: self-attention和残差连接(即最终输出)
    zerollm_backend::memcpy(d_attn_out, d_ln1_out, total_elements * sizeof(float), zerollm_backend::CopyKind::D2D);
    zerollm_backend::memcpy(d_input, d_ln1_out, total_elements * sizeof(float), zerollm_backend::CopyKind::D2D);

    // Backward through self-attention
    // 注意：这里的 d_attn_out 被用作输入和输出。
    // 我们假设 self_attn_->backward 会用计算结果覆盖 d_attn_out。
    // d_attn_out (输入) = dL/d(Attn)
    // d_attn_out (输出) = dL/d(X)_attn
    LOG_DEBUG("Backward through self-attention");
    self_attn_->backward(d_attn_out, d_attn_out);

    // 将注意力的梯度累加到最终输出
    // d_input (总) = d_input (残差) + d_attn_out (来自MHA)
    LOG_DEBUG("Accumulating gradients from residual and self-attention branches");
    add_inplace<float>(d_input, d_attn_out, total_elements, 0);

    // Clean up temporary memory
    zerollm_backend::free(d_ln2_out);
    zerollm_backend::free(d_ff_out);
    zerollm_backend::free(d_ff_in);
    zerollm_backend::free(d_ln1_out);
    zerollm_backend::free(d_attn_out);
    
    // --- 别忘了释放新增的缓冲区 ---
    zerollm_backend::free(d_ff_in_from_ffn);

    // Synchronize and check errors
    zerollm_backend::device_synchronize();
    zerollm_backend::check_last_error("TransformerDecoderBlock::backward() failed");
    LOG_DEBUG("TransformerDecoderBlock backward completed");
}

/**
 * @brief 保存模型参数
 * 
 * @param path 保存路径
 */
void TransformerDecoderBlock::save(const std::string& path) {
    LOG_DEBUG("Saving TransformerDecoderBlock layer to " << path);
    
    // 创建目录
    Serializer::create_directory(path);
    
    // 保存各个子层
    self_attn_->save(path + "/self_attn");
    ln1_->save(path + "/ln1");
    ff_->save(path + "/ff");
    ln2_->save(path + "/ln2");
    
    LOG_DEBUG("TransformerDecoderBlock layer saved successfully");
}

/**
 * @brief 加载模型参数
 * 
 * @param path 加载路径
 */
void TransformerDecoderBlock::load(const std::string& path) {
    LOG_DEBUG("Loading TransformerDecoderBlock layer from " << path);
    
    // 加载各个子层
    self_attn_->load(path + "/self_attn");
    ln1_->load(path + "/ln1");
    ff_->load(path + "/ff");
    ln2_->load(path + "/ln2");
    
    LOG_DEBUG("TransformerDecoderBlock layer loaded successfully");
}



/**
 * @brief 设置优化器
 * 
 * @param config 优化器配置
 */
void TransformerDecoderBlock::set_optimizer(OptimizerConfig config) {
    self_attn_->set_optimizer(config);
    ln1_->set_optimizer(config);
    ln2_->set_optimizer(config);
    ff_->set_optimizer(config);
}

/**
 * @brief 优化器步进
 * 
 * @param learning_rate 学习率
 */
void TransformerDecoderBlock::step(float learning_rate) {
    self_attn_->step(learning_rate);
    ln1_->step(learning_rate);
    ln2_->step(learning_rate);
    ff_->step(learning_rate);
}

/**
 * @brief 清零所有梯度
 */
void TransformerDecoderBlock::zero_grad() {
    self_attn_->zero_grad();
    ln1_->zero_grad();
    ln2_->zero_grad();
    ff_->zero_grad();
}