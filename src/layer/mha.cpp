#include "mha.hpp"
#include <stdexcept>
#include <string>
#include "config.hpp"  // 引入后端配置头文件


#include "mha.cuh"
#include "add.cuh"

#include "async_logger.hpp"
#include "serializer.hpp"

/**
 * @brief 多头注意力构造函数
 * @param embed_dim 输入 embedding 的维度
 * @param num_heads 多头注意力的头数
 * @param with_grad 是否需要梯度计算
 * @param is_causal 是否为因果注意力（用于解码器）
 */
MultiHeadAttention::MultiHeadAttention(int embed_dim, int num_heads, bool with_grad, bool is_causal)
    : embed_dim_(embed_dim),
      num_heads_(num_heads),
      head_dim_(embed_dim / num_heads),
      is_causal_(is_causal),
      last_batch_size_(0),
      last_seq_len_(0),
      Q_(nullptr),
      K_(nullptr),
      V_(nullptr),
      attn_output_(nullptr),
      attention_scores_(nullptr),
      mask_(nullptr) {
    
    if (embed_dim % num_heads != 0) {
        throw std::invalid_argument("embed_dim must be divisible by num_heads");
    }
    
    // Create projection layers
    q_proj_ = new Linear(embed_dim_, embed_dim_, true, with_grad);  // With bias
    k_proj_ = new Linear(embed_dim_, embed_dim_, true, with_grad);
    v_proj_ = new Linear(embed_dim_, embed_dim_, true, with_grad);
    out_proj_ = new Linear(embed_dim_, embed_dim_, true, with_grad);
}

/**
 * @brief 多头注意力析构函数，释放内存
 */
MultiHeadAttention::~MultiHeadAttention() {
    delete q_proj_;
    delete k_proj_;
    delete v_proj_;
    delete out_proj_;
    
    zerollm_backend::free(Q_);
    zerollm_backend::free(K_);
    zerollm_backend::free(V_);
    zerollm_backend::free(attn_output_);
    zerollm_backend::free(attention_scores_);
    zerollm_backend::free(mask_);
}

/**
 * @brief 设置mask
 * @param mask mask [seq_len, seq_len]
 */
void MultiHeadAttention::set_mask(const bool* mask) {
    // 这里可以设置自定义掩码
    // 实现细节取决于具体的掩码格式
}

/**
 * @brief 前向传播
 * 
 * 执行多头自注意力计算:
 * 1. 通过线性投影得到Q, K, V
 * 2. 计算注意力分数: QK^T / sqrt(d_k)
 * 3. 应用掩码(可选)
 * 4. 应用softmax
 * 5. 计算输出: softmax(QK^T/sqrt(d_k)) * V
 * 6. 通过输出投影层
 * @param output 输出结果 [batch_size, seq_len, embed_dim]
 * @param input 输入 [batch_size, seq_len, embed_dim]
 * @param batch_size batch_size
 * @param seq_len seq_len
 */
void MultiHeadAttention::forward(float* output, const float* input, int batch_size, int seq_len) {
    LOG_DEBUG("MultiHeadAttention forward: batch_size=" << batch_size << ", seq_len=" << seq_len 
              << ", embed_dim=" << embed_dim_ << ", num_heads=" << num_heads_);
    
    // Resize buffers if needed
    if (last_batch_size_ != batch_size || last_seq_len_ != seq_len) {
        LOG_DEBUG("Resizing buffers for new batch/seq size");
        zerollm_backend::free(Q_);
        zerollm_backend::free(K_);
        zerollm_backend::free(V_);
        zerollm_backend::free(attn_output_);
        zerollm_backend::free(attention_scores_);
        zerollm_backend::free(mask_);
        
        int total_elements = batch_size * seq_len * embed_dim_; // Q, K, V, attn_output size
        int score_elements = batch_size * num_heads_ * seq_len * seq_len; // attention_scores size
        int mask_elements = seq_len * seq_len; // mask size
        
        Q_ = (float *)zerollm_backend::malloc(total_elements * sizeof(float));
        K_ = (float *)zerollm_backend::malloc(total_elements * sizeof(float));
        V_ = (float *)zerollm_backend::malloc(total_elements * sizeof(float));
        attn_output_ = (float *)zerollm_backend::malloc(total_elements * sizeof(float));
        attention_scores_ = (float *)zerollm_backend::malloc(score_elements * sizeof(float));
        
        // 如果是因果注意力，创建因果掩码
        if (is_causal_) {
            mask_ = (bool *)zerollm_backend::malloc(mask_elements * sizeof(bool));
            
            // 创建因果掩码（下三角矩阵）
            bool* host_mask = new bool[mask_elements];
            for (int i = 0; i < seq_len; ++i) {
                for (int j = 0; j < seq_len; ++j) {
                    host_mask[i * seq_len + j] = (j <= i); // 下三角（包括对角线）为true
                }
            }
            
            zerollm_backend::memcpy(mask_, host_mask, mask_elements * sizeof(bool), zerollm_backend::CopyKind::H2D);
            delete[] host_mask;
        }
        
        last_batch_size_ = batch_size;
        last_seq_len_ = seq_len;
    }
    
    // Project input to Q, K, V，实际上就是对embed_dim进行线性映射，映射前后还是一样的，只是映射的权重矩阵不同
    LOG_DEBUG("Projecting input to Q, K, V");
    q_proj_->forward(Q_, input, batch_size * seq_len);  // [batch_size * seq_len, embed_dim] -> [batch_size * seq_len, embed_dim]
    k_proj_->forward(K_, input, batch_size * seq_len); 
    v_proj_->forward(V_, input, batch_size * seq_len); 

    // Q, K, V 的形状可以看作 [batch_size, seq_len, num_heads, head_dim], 其中 head_dim = embed_dim / num_heads
    LOG_DEBUG("Computing multi-head attention");
    mha_forward<float>(
        Q_, K_, V_, attn_output_, attention_scores_,
        is_causal_ ? mask_ : nullptr, // 使用因果掩码或不使用掩码
        (int64_t)batch_size,
        (int64_t)seq_len,
        (int64_t)num_heads_,
        (int64_t)embed_dim_ / num_heads_,
        is_causal_,
        0  // cudaStream_t stream
    );

    // Apply output projection
    LOG_DEBUG("Applying output projection");
    out_proj_->forward(output, attn_output_, batch_size * seq_len);
    
    // Synchronize and check errors
    zerollm_backend::device_synchronize();
    zerollm_backend::check_last_error("MultiHeadAttention::forward() failed");
    LOG_DEBUG("MultiHeadAttention forward completed");
}

/**
 * @brief 反向传播
 * 
 * 计算多头注意力的梯度
 * @param d_input 输入梯度 [batch_size, seq_len, embed_dim]
 * @param d_output 输出梯度 [batch_size, seq_len, embed_dim]
 */
void MultiHeadAttention::backward(float* d_input, const float* d_output) {
    LOG_DEBUG("MultiHeadAttention backward started");
    
    if (last_batch_size_ == 0 || last_seq_len_ == 0) {
        throw std::runtime_error("Must call forward() before backward()");
    }

    int batch_size = last_batch_size_;
    int seq_len = last_seq_len_;
    int total_elements = batch_size * seq_len * embed_dim_;

    float* d_attn_output = nullptr;
    float* d_Q = nullptr;
    float* d_K = nullptr;
    float* d_V = nullptr;
    float* temp_d_input = nullptr;

    // 分配 GPU 内存
    d_attn_output = (float *)zerollm_backend::malloc(total_elements * sizeof(float));
    d_Q = (float *)zerollm_backend::malloc(total_elements * sizeof(float));
    d_K = (float *)zerollm_backend::malloc(total_elements * sizeof(float));
    d_V = (float *)zerollm_backend::malloc(total_elements * sizeof(float));
    temp_d_input = (float *)zerollm_backend::malloc(total_elements * sizeof(float)); // 复用 buffer

    if (!d_attn_output || !d_Q || !d_K || !d_V || !temp_d_input) {
        // 释放已经分配的（如果有）
        if (d_attn_output) zerollm_backend::free(d_attn_output);
        if (d_Q) zerollm_backend::free(d_Q);
        if (d_K) zerollm_backend::free(d_K);
        if (d_V) zerollm_backend::free(d_V);
        if (temp_d_input) zerollm_backend::free(temp_d_input);
        throw std::runtime_error("MultiHeadAttention::backward(): GPU malloc failed");
    }

    // 1. Backprop through output projection: 得到 d_attn_output
    LOG_DEBUG("Backward through output projection");
    out_proj_->backward(d_attn_output, d_output);

    // 2. MHA backward: 得到 d_Q, d_K, d_V
    LOG_DEBUG("Computing MHA backward");
    mha_backward<float>(
        d_attn_output,
        Q_, K_, V_,
        attention_scores_,
        is_causal_ ? mask_ : nullptr,
        d_Q, d_K, d_V,
        (int64_t)batch_size,
        (int64_t)seq_len,
        (int64_t)num_heads_,
        (int64_t)embed_dim_ / num_heads_,
        is_causal_,
        0  // cudaStream_t stream
    );

    // 3. 把 d_Q, d_K, d_V 投影回输入空间并累加到 d_input
    // (a) q_proj_->backward 会把 d_Q 的投影结果覆盖写入 d_input（即 d_input = d_q_proj）
    LOG_DEBUG("Backward through Q projection");
    q_proj_->backward(d_input, d_Q);

    // (b) 使用 temp_d_input 来存放 k_proj_ 的 backward 输出，然后设备端累加到 d_input
    LOG_DEBUG("Backward through K projection");
    k_proj_->backward(temp_d_input, d_K);
    // 使用你实现的设备端 inplace 加法，将 temp_d_input 加到 d_input 上
    // add_inplace 的签名： template<typename T> void add_inplace(T* a, const T* b, int size, cudaStream_t stream = 0);
    LOG_DEBUG("Accumulating K projection gradients");
    add_inplace<float>(d_input, temp_d_input, total_elements, 0);

    // (c) 使用 temp_d_input 来存放 v_proj_ 的 backward 输出，然后设备端累加到 d_input
    LOG_DEBUG("Backward through V projection");
    v_proj_->backward(temp_d_input, d_V);
    LOG_DEBUG("Accumulating V projection gradients");
    add_inplace<float>(d_input, temp_d_input, total_elements, 0);

    // 释放临时内存
    zerollm_backend::free(d_attn_output);
    zerollm_backend::free(d_Q);
    zerollm_backend::free(d_K);
    zerollm_backend::free(d_V);
    zerollm_backend::free(temp_d_input);

    // Synchronize and check errors
    zerollm_backend::device_synchronize();
    zerollm_backend::check_last_error("MultiHeadAttention::backward() failed");
    LOG_DEBUG("MultiHeadAttention backward completed");
}

/**
 * @brief 保存模型参数
 * 
 * @param path 保存路径
 */
void MultiHeadAttention::save(const std::string& path) {
    LOG_DEBUG("Saving MultiHeadAttention layer to " << path);
    
    // 创建目录
    Serializer::create_directory(path);
    
    // 保存各个子层
    q_proj_->save(path + "/q_proj");
    k_proj_->save(path + "/k_proj");
    v_proj_->save(path + "/v_proj");
    out_proj_->save(path + "/out_proj");
    
    LOG_DEBUG("MultiHeadAttention layer saved successfully");
}

/**
 * @brief 加载模型参数
 * 
 * @param path 加载路径
 */
void MultiHeadAttention::load(const std::string& path) {
    LOG_DEBUG("Loading MultiHeadAttention layer from " << path);
    
    // 加载各个子层
    q_proj_->load(path + "/q_proj");
    k_proj_->load(path + "/k_proj");
    v_proj_->load(path + "/v_proj");
    out_proj_->load(path + "/out_proj");
    
    LOG_DEBUG("MultiHeadAttention layer loaded successfully");
}


/**
 * @brief 设置优化器
 * 
 * @param config 优化器配置
 */
void MultiHeadAttention::set_optimizer(OptimizerConfig config) {
    q_proj_->set_optimizer(config);
    k_proj_->set_optimizer(config);
    v_proj_->set_optimizer(config);
    out_proj_->set_optimizer(config);
}

/**
 * @brief 优化器步进
 * 
 * @param learning_rate 学习率
 */
void MultiHeadAttention::step(float learning_rate) {
    q_proj_->step(learning_rate);
    k_proj_->step(learning_rate);
    v_proj_->step(learning_rate);
    out_proj_->step(learning_rate);
}

/**
 * @brief 清零所有梯度
 */
void MultiHeadAttention::zero_grad() {
    q_proj_->zero_grad();
    k_proj_->zero_grad();
    v_proj_->zero_grad();
    out_proj_->zero_grad();
}
