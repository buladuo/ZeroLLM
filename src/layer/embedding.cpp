// embedding.cpp
#include "embedding.hpp"
#include <cmath>
#include <stdexcept>
#include <random>
#include <cstring> // for memcpy
#include <cmath>   // for sin, cos, pow
#include "config.hpp"  // 引入后端配置头文件
#include "async_logger.hpp"
#include "serializer.hpp"

#include "embedding.cuh"
#include "sgd.hpp"
#include "adam.hpp"
#include "adamw.hpp"

/**
 * @brief Embedding类构造函数
 * @param vocab_size 词汇表大小
 * @param embed_dim 嵌入维度
 * @param max_seq_len 最大序列长度
 * @param with_grad 是否需要梯度计算
 */
Embedding::Embedding(int vocab_size, int embed_dim, int max_seq_len, bool with_grad)
    : vocab_size_(vocab_size),
      embed_dim_(embed_dim),
      max_seq_len_(max_seq_len),
      with_grad_(with_grad),
      pos_encoding_table_(nullptr),
      d_embedding_table_(nullptr),
      d_embedding_table_grad_(nullptr),
      d_pos_encoding_table_(nullptr) {
          
    LOG_DEBUG("Initializing Embedding layer: vocab_size=" << vocab_size << ", embed_dim=" << embed_dim 
              << ", max_seq_len=" << max_seq_len);
    
    // 初始化词嵌入表
    initializeEmbeddingTable();
    
    // 初始化位置编码表
    initializePositionalEncodingTable();
    
    // 分配设备内存
    d_embedding_table_ = (float*)zerollm_backend::malloc((int64_t)vocab_size_ * embed_dim_ * sizeof(float));
    d_pos_encoding_table_ = (float*)zerollm_backend::malloc((int64_t)max_seq_len_ * embed_dim_ * sizeof(float));
    
    // 将嵌入表和位置编码表复制到设备内存
    zerollm_backend::memcpy(d_embedding_table_, embedding_table_, 
                           (int64_t)vocab_size_ * embed_dim_ * sizeof(float), 
                           zerollm_backend::CopyKind::H2D);
    zerollm_backend::memcpy(d_pos_encoding_table_, pos_encoding_table_, 
                           (int64_t)max_seq_len_ * embed_dim_ * sizeof(float), 
                           zerollm_backend::CopyKind::H2D);
    
    // 如果需要梯度计算，分配梯度内存
    if (with_grad_) {
        d_embedding_table_grad_ = (float*)zerollm_backend::malloc((int64_t)vocab_size_ * embed_dim_ * sizeof(float));
        zerollm_backend::memset(d_embedding_table_grad_, 0, (int64_t)vocab_size_ * embed_dim_ * sizeof(float));
    }
    
    // 释放主机内存
    delete[] embedding_table_;
    embedding_table_ = nullptr;
    delete[] pos_encoding_table_;
    pos_encoding_table_ = nullptr;
    
    LOG_DEBUG("Embedding layer initialized");
}

/**
 * @brief Embedding类析构函数，释放CPU和设备内存
 */
Embedding::~Embedding() {

    // 释放设备内存
    zerollm_backend::free(d_embedding_table_);
    zerollm_backend::free(d_pos_encoding_table_);
    
    if (d_embedding_table_grad_) {
        zerollm_backend::free(d_embedding_table_grad_);
    }
}

/**
 * @brief 初始化嵌入表
 * 
 * 使用简单的随机初始化方法初始化词汇嵌入表
 */
void Embedding::initializeEmbeddingTable() {
    // 创建嵌入表
    embedding_table_ = new float[vocab_size_ * embed_dim_];
    
    // 使用简单的随机初始化（实际应用中可能会使用更复杂的初始化方法）
    for (int i = 0; i < vocab_size_ * embed_dim_; ++i) {
        embedding_table_[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX) * 0.02f - 0.01f;
    }
}

/**
 * @brief 初始化位置编码表
 * 
 * 使用正弦位置编码公式初始化位置编码表
 * PE(pos, 2i) = sin(pos / 10000^(2i / d_model))
 * PE(pos, 2i+1) = cos(pos / 10000^(2i / d_model))
 */
void Embedding::initializePositionalEncodingTable() {
    // 创建位置编码表
    pos_encoding_table_ = new float[max_seq_len_ * embed_dim_];
    
    // 实现正弦位置编码
    for (int pos = 0; pos < max_seq_len_; ++pos) {
        for (int i = 0; i < embed_dim_; ++i) {
            float angle_rate;
            int div_term_idx = i / 2;  // 对于位置 i，使用 i/2 作为指数
            angle_rate = static_cast<float>(pos) / pow(10000.0f, static_cast<float>(2 * div_term_idx) / embed_dim_);
            if (i % 2 == 0) {
                // 偶数维度使用sin
                pos_encoding_table_[pos * embed_dim_ + i] = sin(angle_rate);
            } else {
                // 奇数维度使用cos
                pos_encoding_table_[pos * embed_dim_ + i] = cos(angle_rate);
            }
        }
    }
}

/**
 * @brief 前向传播函数
 * 
 * 将输入的token IDs转换为嵌入向量表示，包含词嵌入和位置编码
 * @param output 输出嵌入向量 [batch_size, seq_len, embed_dim]
 * @param input 输入token IDs [batch_size, seq_len]
 * @param batch_size 批处理大小
 * @param seq_len 序列长度
 */
void Embedding::forward(float* output, const int* input, int batch_size, int seq_len) {
    LOG_DEBUG("Embedding forward: batch_size=" << batch_size << ", seq_len=" << seq_len);
    
    if (seq_len > max_seq_len_) {
        throw std::runtime_error("Sequence length exceeds maximum allowed length");
    }
    
    // 调用CUDA kernel进行前向传播
    LOG_DEBUG("Calling CUDA kernel for embedding forward");
    cuda_embedding_forward<float, int>(
        input,
        d_embedding_table_,
        d_pos_encoding_table_,
        output,
        batch_size,
        seq_len,
        embed_dim_,
        vocab_size_,
        max_seq_len_,
        0  // cudaStream_t stream
    );
    zerollm_backend::device_synchronize();
    zerollm_backend::check_last_error("Embedding::forward failed");
    LOG_DEBUG("Embedding forward completed");
}

/**
 * @brief 反向传播函数
 * 
 * 计算嵌入层的梯度
 * @param input 输入token IDs [batch_size, seq_len]
 * @param d_output 输出梯度 [batch_size, seq_len, embed_dim]
 * @param batch_size 批处理大小
 * @param seq_len 序列长度
 */
void Embedding::backward(const int* input, const float* d_output, int batch_size, int seq_len, bool accumulate) {
    LOG_DEBUG("Embedding backward: batch_size=" << batch_size << ", seq_len=" << seq_len 
              << ", accumulate=" << accumulate);
    
    if (!with_grad_) {
        LOG_DEBUG("Gradient computation disabled, skipping backward pass");
        return; // 如果不需要梯度，则直接返回
    }
    
    // 如果不需要累加，则先清零梯度
    if (!accumulate) {
        LOG_DEBUG("Zeroing gradients");
        zerollm_backend::memset(d_embedding_table_grad_, 0, vocab_size_ * embed_dim_ * sizeof(float));
    }

    // 调用CUDA kernel进行反向传播
    LOG_DEBUG("Calling CUDA kernel for embedding backward");
    cuda_embedding_backward<float, int>(
        input,
        d_output,
        d_embedding_table_grad_,
        batch_size,
        seq_len,
        embed_dim_,
        vocab_size_,
        0  // cudaStream_t stream
    );

    
    zerollm_backend::device_synchronize();
    zerollm_backend::check_last_error("Embedding::backward failed");
    LOG_DEBUG("Embedding backward completed");
}
/**
 * @brief 初始化优化器
 * 
 * @param optimizer 优化器对象
 */
void Embedding::set_optimizer(OptimizerConfig config){
    switch (config.type)
    {
    case OptimizerType::SGD:
        this->optimizer_ = new SGD(config.momentum);
        break;
    case OptimizerType::Adam:
        this->optimizer_ = new Adam(config.beta1, config.beta2, config.eps);
        break;
    case OptimizerType::AdamW:
        this->optimizer_ = new AdamW(config.weight_decay, config.beta1, config.beta2, config.eps);
        break;
    default:
        throw std::runtime_error("Invalid optimizer type");
        break;
    };
}
/**
 * @brief 优化器步进
 * 
 * @param learning_rate 学习率
 */
void Embedding::step(float learning_rate) {
    if (with_grad_ && optimizer_) {
        optimizer_->step(d_embedding_table_, d_embedding_table_grad_, vocab_size_ * embed_dim_, learning_rate);
    }else{
        throw std::runtime_error("Embedding::step failed: no optimizer or no gradient");
    }
}

/**
 * @brief 清零梯度
 */
void Embedding::zero_grad() {
    if (with_grad_ && d_embedding_table_grad_) {
        zerollm_backend::memset(d_embedding_table_grad_, 0, vocab_size_ * embed_dim_ * sizeof(float));
    }
}


/**
 * @brief 保存模型参数
 * 
 * @param path 保存路径
 */
void Embedding::save(const std::string& path) {
    LOG_DEBUG("Saving Embedding layer to " << path);
    
    // 创建目录
    Serializer::create_directory(path);
    Serializer::create_directory(path + "/weights");
    
    // 保存嵌入表
    Serializer::save_tensor(d_embedding_table_, (size_t)vocab_size_ * embed_dim_, path + "/weights/table.bin");
    
    LOG_DEBUG("Embedding layer saved successfully");
}

/**
 * @brief 加载模型参数
 * 
 * @param path 加载路径
 */
void Embedding::load(const std::string& path) {
    LOG_DEBUG("Loading Embedding layer from " << path);
    
    // 检查文件是否存在
    std::string weight_file = path + "/weights/table.bin";
    std::ifstream file(weight_file, std::ios::binary);
    if (!file.is_open()) {
        LOG_ERROR("Failed to open embedding weight file: " << weight_file);
        throw std::runtime_error("Failed to open embedding weight file: " + weight_file);
    }
    file.close();
    
    // 加载嵌入表
    Serializer::load_tensor(d_embedding_table_, (size_t)vocab_size_ * embed_dim_, weight_file);
    
    LOG_DEBUG("Embedding layer loaded successfully");
}

/**
 * @brief 获取嵌入表梯度指针
 * @return 嵌入表梯度指针
 */
float* Embedding::d_embedding_table() {
    return d_embedding_table_grad_;
}

float* Embedding::embedding_table_device() {
    return d_embedding_table_;
}

const float* Embedding::embedding_table_device() const {
    return d_embedding_table_;
}