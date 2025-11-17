// include/linear_layer.hpp
#include <stdexcept>
#include <string>
#include <random>
#include <vector>


#include "linear.hpp"
#include "config.hpp"
#include "linear.cuh"

#include "sgd.hpp"
#include "adam.hpp"
#include "adamw.hpp"

#include "async_logger.hpp"
#include "serializer.hpp"


/**
 * @brief 线性层构造函数
 * @param in_features 输入特征数
 * @param out_features 输出特征数
 * @param use_bias 是否使用偏置项
 * @param with_grad 是否需要梯度计算
 */
Linear::Linear(int in_features, int out_features, bool use_bias, bool with_grad)
    : in_features_(in_features),
      out_features_(out_features),
      use_bias_(use_bias),
      with_grad_(with_grad),
      weight_(nullptr),
      bias_(nullptr),
      d_weight_(nullptr),
      d_bias_(nullptr),
      input_(nullptr),
      last_batch_size_(0)
{

    weight_ = (float*)zerollm_backend::malloc((int64_t)out_features_ * in_features_ * sizeof(float));
    initializeWeights(); // Xavier 初始化

    if (use_bias_) {
        bias_ = (float*)zerollm_backend::malloc((int64_t)out_features_ * sizeof(float));
        zerollm_backend::memset(bias_, 0, (int64_t)out_features_ * sizeof(float));
    }

    if (with_grad_) {
        d_weight_ = (float*)zerollm_backend::malloc((int64_t)out_features_ * in_features_ * sizeof(float));
        zerollm_backend::memset(d_weight_, 0, (int64_t)out_features_ * in_features_ * sizeof(float));

        if (use_bias_) {
            d_bias_ = (float*)zerollm_backend::malloc((int64_t)out_features_ * sizeof(float));
            zerollm_backend::memset(d_bias_, 0, (int64_t)out_features_ * sizeof(float));
        }
    }

    register_parameter("weight", weight_, d_weight_, (size_t)out_features_ * in_features_, with_grad_);
    register_parameter("bias", bias_, d_bias_, (size_t)out_features_, with_grad_);
}


/**
 * @brief 初始化权重
 * 
 * 使用Xavier初始化方法（均匀分布）初始化权重
 */
void Linear::initializeWeights() {
    // 在主机上生成随机权重
    std::vector<float> host_weights(out_features_ * in_features_);
    
    float scale = sqrtf(6.0f / (in_features_ + out_features_));
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-scale, scale);
    
    for (int i = 0; i < out_features_ * in_features_; ++i) {
        host_weights[i] = dis(gen);
    }

    // 将随机权重复制到设备内存（自动选择正确后端）
    zerollm_backend::memcpy(
        weight_,
        host_weights.data(),
        out_features_ * in_features_ * sizeof(float),
        zerollm_backend::CopyKind::H2D
    );
}

/**
 * @brief 线性层析构函数，释放GPU内存
 */
Linear::~Linear() {
    if (weight_) zerollm_backend::free(weight_);
    if (bias_) zerollm_backend::free(bias_);
    if (d_weight_) zerollm_backend::free(d_weight_);
    if (d_bias_) zerollm_backend::free(d_bias_);
}

/**
 * @brief 清零梯度
 */
void Linear::zero_grad() {
    if (with_grad_) {
        zerollm_backend::memset(d_weight_, 0, (int64_t)out_features_ * in_features_ * sizeof(float));
        if (use_bias_) {
            zerollm_backend::memset(d_bias_, 0, (int64_t)out_features_ * sizeof(float));
        }
    }
}


/**
 * @brief 前向传播
 * 
 * 执行线性变换: Y = X * W^T + b
 * @param output 输出数据 [batch_size, out_features]
 * @param input 输入数据 [batch_size, in_features]
 * @param batch_size 批次大小
 */
void Linear::forward(float* output, const float* input, int batch_size) {
    LOG_DEBUG("Linear forward: batch_size=" << batch_size << ", in_features=" << in_features_ 
              << ", out_features=" << out_features_);
    
    if (with_grad_) {
        this->input_ = input;
        this->last_batch_size_ = batch_size;
    }

    linear_forward<float>(
        input,
        weight_,
        use_bias_ ? bias_ : nullptr,
        output,
        (int64_t)batch_size,
        (int64_t)in_features_,
        (int64_t)out_features_,
        use_bias_,
        0  // cudaStream_t stream
    );

    zerollm_backend::device_synchronize();
    zerollm_backend::check_last_error("Linear forward failed");
    LOG_DEBUG("Linear forward completed");
}

/**
 * @brief 反向传播
 * 
 * 计算线性层的梯度
 * @param d_input 输入梯度 [batch_size, in_features]
 * @param d_output 输出梯度 [batch_size, out_features]
 */
void Linear::backward(float* d_input, const float* d_output) {
    LOG_DEBUG("Linear backward: batch_size=" << last_batch_size_ << ", in_features=" << in_features_ 
              << ", out_features=" << out_features_);
    
    if (!with_grad_) {
        throw std::runtime_error("Cannot call backward() when with_grad=false");
    }
    if (this->input_ == nullptr) {
        throw std::runtime_error("Must call forward() before backward()");
    }

    linear_backward<float>(
        d_output,
        input_,
        weight_,
        d_input,
        d_weight_,
        use_bias_ ? d_bias_ : nullptr,
        (int64_t)last_batch_size_,
        (int64_t)in_features_,
        (int64_t)out_features_,
        use_bias_,
        0  // cudaStream_t stream
    );

    zerollm_backend::device_synchronize();
    zerollm_backend::check_last_error("Linear::backward() failed");
    LOG_DEBUG("Linear backward completed");
}

/**
 * @brief 设置优化器
 * 
 * @param optimizer 优化器
 */
void Linear::set_optimizer(OptimizerConfig config) {
    switch (config.type)
    {
    case OptimizerType::SGD:
        this->weight_optimizer_ = new SGD(config.momentum);
        this->bias_optimizer_ = new SGD(config.momentum);
        break;
    case OptimizerType::Adam:
        this->weight_optimizer_ = new Adam(config.beta1, config.beta2, config.eps);
        this->bias_optimizer_ = new Adam(config.beta1, config.beta2, config.eps);
        break;
    case OptimizerType::AdamW:
        this->weight_optimizer_ = new AdamW(config.weight_decay, config.beta1, config.beta2, config.eps);
        this->bias_optimizer_ = new AdamW(config.weight_decay, config.beta1, config.beta2, config.eps);
        break;
    default:
        throw std::runtime_error("Invalid optimizer type");
        break;
    };
}

/**
 * @brief 保存模型参数
 * 
 * @param path 保存路径
 */
void Linear::save(const std::string& path) {
    LOG_DEBUG("Saving Linear layer to " << path);
    
    // 创建目录
    Serializer::create_directory(path);
    Serializer::create_directory(path + "/weights");
    
    // 保存权重
    Serializer::save_tensor(weight_, (size_t)out_features_ * in_features_, path + "/weights/weight.bin");
    
    // 保存偏置（如果使用）
    if (use_bias_) {
        Serializer::save_tensor(bias_, out_features_, path + "/weights/bias.bin");
    }
    
    LOG_DEBUG("Linear layer saved successfully");
}

/**
 * @brief 加载模型参数
 * 
 * @param path 加载路径
 */
void Linear::load(const std::string& path) {
    LOG_DEBUG("Loading Linear layer from " << path);
    
    // 加载权重
    Serializer::load_tensor(weight_, (size_t)out_features_ * in_features_, path + "/weights/weight.bin");
    
    // 加载偏置（如果使用）
    if (use_bias_) {
        Serializer::load_tensor(bias_, out_features_, path + "/weights/bias.bin");
    }
    
    LOG_DEBUG("Linear layer loaded successfully");
}

/**
 * @brief 优化权重
 * 
 * @param learning_rate 学习率
 */
void Linear::step(float learning_rate) {
    if (with_grad_ && weight_optimizer_ && bias_optimizer_) {
        weight_optimizer_->step(weight_, d_weight_, out_features_ * in_features_, learning_rate);
        bias_optimizer_->step(bias_, d_bias_, out_features_, learning_rate);
    }else{
        throw std::runtime_error("Linear step() no optimizer or no gradient");
    }
}

/**
 * @brief 获取权重指针
 * @return 权重指针
 */
float* Linear::weight() { return weight_; }

/**
 * @brief 获取偏置指针
 * @return 偏置指针
 */
float* Linear::bias() { return bias_; }

/**
 * @brief 获取权重梯度指针
 * @return 权重梯度指针
 */
float* Linear::d_weight() { return d_weight_; }

/**
 * @brief 获取偏置梯度指针
 * @return 偏置梯度指针
 */
float* Linear::d_bias() { return d_bias_; }