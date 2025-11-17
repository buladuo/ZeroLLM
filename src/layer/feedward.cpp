#include "feedward.hpp"
#include <stdexcept>
#include <string>
#include "config.hpp"  // 引入后端配置头文件
#include "async_logger.hpp"
#include "serializer.hpp"

/**
 * @brief 前馈神经网络构造函数
 * @param embed_dim 输入嵌入维度
 * @param ff_hidden_dim 前馈网络隐藏层维度
 * @param with_grad 是否需要梯度计算
 */
FeedForward::FeedForward(int embed_dim, int ff_hidden_dim, bool with_grad)
    : embed_dim_(embed_dim),
      ff_hidden_dim_(ff_hidden_dim),
      ff1_(nullptr),
      relu_(nullptr),
      ff2_(nullptr),
      ff1_output_(nullptr),
      relu_output_(nullptr),
      ff2_output_(nullptr),
      last_batch_size_(0),
      last_seq_len_(0) {
          
    LOG_DEBUG("Initializing FeedForward: embed_dim=" << embed_dim << ", ff_hidden_dim=" << ff_hidden_dim);
    
    ff1_ = new Linear(embed_dim, ff_hidden_dim, true, with_grad);  // with bias
    relu_ = new ReLU();
    ff2_ = new Linear(ff_hidden_dim, embed_dim, true, with_grad);  // with bias
    
    register_module("ff1", ff1_);
    register_module("relu", ff2_);

    LOG_DEBUG("FeedForward initialized");
}

/**
 * @brief 前馈网络析构函数，释放内存
 */
FeedForward::~FeedForward() {
    delete ff1_;
    delete relu_;
    delete ff2_;
    
    zerollm_backend::free(ff1_output_);
    zerollm_backend::free(relu_output_);
    zerollm_backend::free(ff2_output_);
}

/**
 * @brief 前向传播
 * 
 * 执行前馈神经网络计算:
 * 输入 -> Linear1 -> ReLU -> Linear2 -> 输出
 * @param output 输出数据 [batch_size, seq_len, embed_dim]
 * @param input 输入数据 [batch_size, seq_len, embed_dim]
 * @param batch_size 批次大小
 * @param seq_len 序列长度
 */
void FeedForward::forward(float* output, const float* input, int batch_size, int seq_len) {
    LOG_DEBUG("FeedForward forward: batch_size=" << batch_size << ", seq_len=" << seq_len 
              << ", embed_dim=" << embed_dim_ << ", ff_hidden_dim=" << ff_hidden_dim_);
    
    // 重新分配缓冲区
    if (last_batch_size_ != batch_size || last_seq_len_ != seq_len) {
        LOG_DEBUG("Resizing buffers for new batch/seq size");
        zerollm_backend::free(ff1_output_);
        zerollm_backend::free(relu_output_);
        zerollm_backend::free(ff2_output_);
        
        int total_elements = batch_size * seq_len * embed_dim_;
        int hidden_elements = batch_size * seq_len * ff_hidden_dim_;
        
        ff1_output_ = (float*)zerollm_backend::malloc(hidden_elements * sizeof(float));
        relu_output_ = (float*)zerollm_backend::malloc(hidden_elements * sizeof(float));
        ff2_output_ = (float*)zerollm_backend::malloc(total_elements * sizeof(float));
        
        last_batch_size_ = batch_size;
        last_seq_len_ = seq_len;
    }
    
    // Forward through first linear layer
    LOG_DEBUG("Forward through first linear layer");
    ff1_->forward(ff1_output_, input, batch_size * seq_len);
    
    // Forward through ReLU
    LOG_DEBUG("Forward through ReLU activation");
    relu_->forward(relu_output_, ff1_output_, batch_size * seq_len, ff_hidden_dim_);
    
    // Forward through second linear layer
    LOG_DEBUG("Forward through second linear layer");
    ff2_->forward(output, relu_output_, batch_size * seq_len);
    
    zerollm_backend::device_synchronize();
    zerollm_backend::check_last_error("FeedForward::forward() failed");
    LOG_DEBUG("FeedForward forward completed");
}

/**
 * @brief 反向传播
 *
 * 计算前馈神经网络的梯度
 * @param d_input 输入梯度 [batch_size, seq_len, embed_dim]
 * @param d_output 输出梯度 [batch_size, seq_len, embed_dim]
 */
void FeedForward::backward(float* d_input, const float* d_output) {
    LOG_DEBUG("FeedForward backward started");
    
    if (last_batch_size_ == 0 || last_seq_len_ == 0) {
        throw std::runtime_error("Must call forward() before backward()");
    }
    
    int batch_size = last_batch_size_;
    int seq_len = last_seq_len_;
    
    float* d_relu_output = (float*)zerollm_backend::malloc((int64_t)batch_size * seq_len * ff_hidden_dim_ * sizeof(float));
    float* d_ff1_output = (float*)zerollm_backend::malloc((int64_t)batch_size * seq_len * ff_hidden_dim_ * sizeof(float));
    float* d_ff1_input = (float*)zerollm_backend::malloc((int64_t)batch_size * seq_len * embed_dim_ * sizeof(float));
    
    // Backward through second linear layer
    LOG_DEBUG("Backward through second linear layer");
    ff2_->backward(d_relu_output, d_output);
    
    // Backward through ReLU
    LOG_DEBUG("Backward through ReLU activation");
    relu_->backward(d_ff1_output, d_relu_output);
    
    // Backward through first linear layer
    LOG_DEBUG("Backward through first linear layer");
    ff1_->backward(d_input, d_ff1_output);
    
    // Clean up
    zerollm_backend::free(d_relu_output);
    zerollm_backend::free(d_ff1_output);
    zerollm_backend::free(d_ff1_input);
    
    zerollm_backend::device_synchronize();
    zerollm_backend::check_last_error("FeedForward::backward() failed");
    LOG_DEBUG("FeedForward backward completed");
}

/**
 * @brief 保存模型参数
 * 
 * @param path 保存路径
 */
void FeedForward::save(const std::string& path) {
    LOG_DEBUG("Saving FeedForward layer to " << path);
    
    // 创建目录
    Serializer::create_directory(path);
    
    // 保存各个子层
    ff1_->save(path + "/ff1");
    ff2_->save(path + "/ff2");
    
    LOG_DEBUG("FeedForward layer saved successfully");
}

/**
 * @brief 加载模型参数
 * 
 * @param path 加载路径
 */
void FeedForward::load(const std::string& path) {
    LOG_DEBUG("Loading FeedForward layer from " << path);
    
    // 加载各个子层
    ff1_->load(path + "/ff1");
    ff2_->load(path + "/ff2");
    
    LOG_DEBUG("FeedForward layer loaded successfully");
}

/**
 * @brief 设置优化器
 * 
 * @param config 优化器配置
 */
void FeedForward::set_optimizer(OptimizerConfig config) {
    ff1_->set_optimizer(config);
    ff2_->set_optimizer(config);
}


/**
 * @brief 优化器步进
 * 
 * @param learning_rate 学习率
 */
void FeedForward::step(float learning_rate) {
    ff1_->step(learning_rate);
    ff2_->step(learning_rate);
}

/**
 * @brief 清零所有梯度
 */
void FeedForward::zero_grad() {
    ff1_->zero_grad();
    relu_->zero_grad();
    ff2_->zero_grad();
}