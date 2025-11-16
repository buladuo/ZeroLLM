// layernorm.cpp
#include "layernorm.hpp"
#include <stdexcept>
#include <string>
#include <random>
#include <vector>
#include "config.hpp"  // 引入后端配置头文件
#include "async_logger.hpp"
#include "serializer.hpp"

#include "layernorm.cuh"
#include "sgd.hpp"
#include "adam.hpp"
#include "adamw.hpp"

/**
 * @brief LayerNorm构造函数
 * @param feature_size 特征维度大小
 * @param with_grad 是否需要梯度计算
 * @param eps 防止除零的小常数
 */
LayerNorm::LayerNorm(int feature_size, bool with_grad, float eps)
    : feature_size_(feature_size),
      with_grad_(with_grad),
      input_(nullptr),
      last_batch_size_(0),
      eps_(eps),
      gamma_(nullptr),
      beta_(nullptr),
      d_gamma_(nullptr),
      d_beta_(nullptr),
      mean_(nullptr),
      rstd_(nullptr) {
    
    // 分配参数内存
    gamma_ = (float *)zerollm_backend::malloc(feature_size_ * sizeof(float));
    beta_ = (float *)zerollm_backend::malloc(feature_size_ * sizeof(float));
    
    // 初始化参数
    // gamma 初始化为1，beta 初始化为0
    std::vector<float> host_gamma(feature_size_, 1.0f);
    std::vector<float> host_beta(feature_size_, 0.0f);
    
    zerollm_backend::memcpy(gamma_, host_gamma.data(), feature_size_ * sizeof(float), zerollm_backend::CopyKind::H2D);
    zerollm_backend::memcpy(beta_, host_beta.data(), feature_size_ * sizeof(float), zerollm_backend::CopyKind::H2D);
    
    if (with_grad_) {
        d_gamma_ = (float *)zerollm_backend::malloc(feature_size_ * sizeof(float));
        d_beta_ = (float *)zerollm_backend::malloc(feature_size_ * sizeof(float));
        
        zerollm_backend::memset(d_gamma_, 0, feature_size_ * sizeof(float));
        zerollm_backend::memset(d_beta_, 0, feature_size_ * sizeof(float));
    }
}

/**
 * @brief LayerNorm析构函数，释放GPU内存
 */
LayerNorm::~LayerNorm() {
    zerollm_backend::free(gamma_);
    zerollm_backend::free(beta_);
    zerollm_backend::free(d_gamma_);
    zerollm_backend::free(d_beta_);
    zerollm_backend::free(mean_);
    zerollm_backend::free(rstd_);
}

/**
 * @brief 清零梯度
 */
void LayerNorm::zero_grad() {
    if (with_grad_) {
        zerollm_backend::memset(d_gamma_, 0, feature_size_ * sizeof(float));
        zerollm_backend::memset(d_beta_, 0, feature_size_ * sizeof(float));
    }
}

/**
 * @brief 前向传播
 * 
 * 对输入数据进行层归一化处理
 * @param output 输出数据 [batch_size, feature_size]
 * @param input 输入数据 [batch_size, feature_size]
 * @param batch_size 批次大小
 */
void LayerNorm::forward(float* output, const float* input, int batch_size) {
    LOG_DEBUG("LayerNorm forward: batch_size=" << batch_size << ", feature_size=" << feature_size_);
    
    float* mean_ptr = nullptr;
    float* rstd_ptr = nullptr;
    
    if (with_grad_) {
        this->input_ = input;
        this->last_batch_size_ = batch_size;
        
        // 重新分配或调整中间变量内存大小
        zerollm_backend::free(mean_);
        zerollm_backend::free(rstd_);
        
        mean_ = (float *)zerollm_backend::malloc(batch_size * sizeof(float));
        rstd_ = (float *)zerollm_backend::malloc(batch_size * sizeof(float));
        
        mean_ptr = mean_;
        rstd_ptr = rstd_;
    }
    
    cuda_layernorm_forward<float>(
        input,
        gamma_,
        beta_,
        output,
        mean_ptr,
        rstd_ptr,
        (int64_t)batch_size,
        (int64_t)feature_size_,
        eps_,
        0  // cudaStream_t stream
    );

    
    zerollm_backend::device_synchronize();
    zerollm_backend::check_last_error("LayerNorm::forward() failed");
    LOG_DEBUG("LayerNorm forward completed");
}

/**
 * @brief 反向传播
 * 
 * 计算层归一化的梯度
 * @param d_input 输入梯度 [batch_size, feature_size]
 * @param d_output 输出梯度 [batch_size, feature_size]
 */
void LayerNorm::backward(float* d_input, const float* d_output) {
    LOG_DEBUG("LayerNorm backward: batch_size=" << last_batch_size_ << ", feature_size=" << feature_size_);
    
    if (!with_grad_) {
        throw std::runtime_error("Cannot call backward() when with_grad=false");
    }
    if (this->input_ == nullptr) {
        throw std::runtime_error("Must call forward() before backward()");
    }

    // 累积梯度到 d_gamma_ 和 d_beta_
    cuda_layernorm_backward<float>(
        input_,
        d_output,
        gamma_,
        mean_,
        rstd_,
        d_input,
        d_gamma_,
        d_beta_,
        (int64_t)last_batch_size_,
        (int64_t)feature_size_,
        0  // cudaStream_t stream
    );

    
    zerollm_backend::device_synchronize();
    zerollm_backend::check_last_error("LayerNorm::backward() failed");
    LOG_DEBUG("LayerNorm backward completed");
}


/**
 * @brief 保存模型参数
 * 
 * @param path 保存路径
 */
void LayerNorm::save(const std::string& path) {
    LOG_DEBUG("Saving LayerNorm layer to " << path);
    
    // 创建目录
    Serializer::create_directory(path);
    Serializer::create_directory(path + "/weights");
    
    // 保存gamma和beta参数
    Serializer::save_tensor(gamma_, feature_size_, path + "/weights/gamma.bin");
    Serializer::save_tensor(beta_, feature_size_, path + "/weights/beta.bin");
    
    LOG_DEBUG("LayerNorm layer saved successfully");
}

/**
 * @brief 加载模型参数
 * 
 * @param path 加载路径
 */
void LayerNorm::load(const std::string& path) {
    LOG_DEBUG("Loading LayerNorm layer from " << path);
    
    // 加载gamma和beta参数
    Serializer::load_tensor(gamma_, feature_size_, path + "/weights/gamma.bin");
    Serializer::load_tensor(beta_, feature_size_, path + "/weights/beta.bin");
    
    LOG_DEBUG("LayerNorm layer loaded successfully");
}


void LayerNorm::set_optimizer(OptimizerConfig config){
    switch (config.type)
    {
    case OptimizerType::SGD:
        this->gamma_optimizer_ = new SGD(config.momentum);
        this->beta_optimizer_ = new SGD(config.momentum);
        break;
    case OptimizerType::Adam:
        this->gamma_optimizer_ = new Adam(config.beta1, config.beta2, config.eps);
        this->beta_optimizer_ = new Adam(config.beta1, config.beta2, config.eps);
        break;
    case OptimizerType::AdamW:
        this->gamma_optimizer_ = new AdamW(config.weight_decay, config.beta1, config.beta2, config.eps);
        this->beta_optimizer_ = new AdamW(config.weight_decay, config.beta1, config.beta2, config.eps);
        break;
    default:
        throw std::runtime_error("Invalid optimizer type");
        break;
    };
}

void LayerNorm::step(float learning_rate) {
    this->gamma_optimizer_->step(gamma_, d_gamma_, feature_size_, learning_rate);
    this->beta_optimizer_->step(beta_, d_beta_, feature_size_, learning_rate);
}

/**
 * @brief 获取gamma参数指针
 * @return gamma参数指针
 */
float* LayerNorm::gamma() { 
    return gamma_; 
}

/**
 * @brief 获取beta参数指针
 * @return beta参数指针
 */
float* LayerNorm::beta() { 
    return beta_; 
}

/**
 * @brief 获取gamma梯度指针
 * @return gamma梯度指针
 */
float* LayerNorm::d_gamma() { 
    return d_gamma_; 
}

/**
 * @brief 获取beta梯度指针
 * @return beta梯度指针
 */
float* LayerNorm::d_beta() { 
    return d_beta_; 
}