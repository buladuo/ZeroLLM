#include "transformer_decoder.hpp"
#include <stdexcept>
#include <string>
#include "config.hpp"  // 引入后端配置头文件
#include "async_logger.hpp"
#include "serializer.hpp"

/**
 * @brief Transformer解码器构造函数
 * @param num_layers 解码器层数
 * @param embed_dim 嵌入维度
 * @param num_heads 头的数量
 * @param ff_hidden_dim 前向传播隐藏层的维度
 * @param with_grad 是否需要梯度计算
 */
TransformerDecoder::TransformerDecoder(int num_layers, int embed_dim, int num_heads, int ff_hidden_dim, bool with_grad)
    : num_layers_(num_layers),
      embed_dim_(embed_dim),
      num_heads_(num_heads),
      ff_hidden_dim_(ff_hidden_dim),
      last_batch_size_(0),
      last_seq_len_(0),
      layers_output_(nullptr) {
    
    LOG_DEBUG("Initializing TransformerDecoder: num_layers=" << num_layers << ", embed_dim=" << embed_dim 
              << ", num_heads=" << num_heads << ", ff_hidden_dim=" << ff_hidden_dim);
    
    // 创建解码器层
    for (int i = 0; i < num_layers; ++i) {
        layers_.push_back(new TransformerDecoderBlock(embed_dim, num_heads, ff_hidden_dim, with_grad));
    }
    
    LOG_DEBUG("TransformerDecoder initialized with " << layers_.size() << " layers");
}

/**
 * @brief Transformer解码器析构函数，释放内存
 */
TransformerDecoder::~TransformerDecoder() {
    for (auto layer : layers_) {
        delete layer;
    }
    
    zerollm_backend::free(layers_output_);
}

/**
 * @brief 前向传播
 * 
 * 执行Transformer解码器的前向计算，逐层处理输入
 * @param output 输出 [batch_size, seq_len, embed_dim]
 * @param input 输入 [batch_size, seq_len, embed_dim]
 * @param batch_size batch_size
 * @param seq_len seq_len
 */
void TransformerDecoder::forward(float* output, const float* input, int batch_size, int seq_len) {
    LOG_DEBUG("TransformerDecoder forward: batch_size=" << batch_size << ", seq_len=" << seq_len 
              << ", num_layers=" << num_layers_);
    
    // 创建缓冲区
    if (last_batch_size_ != batch_size || last_seq_len_ != seq_len) {
        LOG_DEBUG("Resizing layers_output buffer");
        zerollm_backend::free(layers_output_);
        
        int total_elements = batch_size * seq_len * embed_dim_;
        layers_output_ = (float *)zerollm_backend::malloc(total_elements * sizeof(float));
        
        last_batch_size_ = batch_size;
        last_seq_len_ = seq_len;
    }
    
    const float* current_input = input;
    
    // 逐层前向传播
    for (int i = 0; i < num_layers_; ++i) {
        LOG_DEBUG("Forward through layer " << (i+1) << "/" << num_layers_);
        // 如果是最后一层，直接输出到output，否则输出到中间缓冲区
        float* current_output = (i == num_layers_ - 1) ? output : layers_output_;
        layers_[i]->forward(current_output, current_input, batch_size, seq_len);
        current_input = current_output;
    }
    
    zerollm_backend::device_synchronize();
    zerollm_backend::check_last_error("TransformerDecoder::forward() failed");
    LOG_DEBUG("TransformerDecoder forward completed");
}

/**
 * @brief 反向传播
 *
 * 反向通过所有解码器层计算梯度
 * @param d_input 输入梯度 [batch_size, seq_len, embed_dim]
 * @param d_output 输出梯度 [batch_size, seq_len, embed_dim]
 */
void TransformerDecoder::backward(float* d_input, const float* d_output) {
    LOG_DEBUG("TransformerDecoder backward started");
    
    if (last_batch_size_ == 0 || last_seq_len_ == 0) {
        throw std::runtime_error("Must call forward() before backward()");
    }
    
    int batch_size = last_batch_size_;
    int seq_len = last_seq_len_;
    int total_elements = batch_size * seq_len * embed_dim_;
    
    // 创建临时梯度缓冲区
    float* d_current = nullptr;
    float* d_temp = nullptr;
    
    d_current = (float *)zerollm_backend::malloc(total_elements * sizeof(float));
    d_temp = (float *)zerollm_backend::malloc(total_elements * sizeof(float));
    
    // 从最后一层开始反向传播
    // 首先将输出梯度复制到当前梯度缓冲区
    LOG_DEBUG("Copying output gradients to current buffer");
    zerollm_backend::memcpy(d_current, d_output, total_elements * sizeof(float), zerollm_backend::CopyKind::D2D);
    
    // 反向遍历所有层
    for (int i = num_layers_ - 1; i >= 0; --i) {
        LOG_DEBUG("Backward through layer " << (i+1) << "/" << num_layers_);
        // 如果是第一层，将梯度写入d_input，否则写入d_temp
        float* d_layer_input = (i == 0) ? d_input : d_temp;
        
        // 执行当前层的反向传播
        layers_[i]->backward(d_layer_input, d_current);
        
        // 交换缓冲区指针，为下一层准备
        if (i > 0) {
            std::swap(d_current, d_temp);
        }
    }
    
    // 清理临时内存
    zerollm_backend::free(d_current);
    zerollm_backend::free(d_temp);
    
    zerollm_backend::device_synchronize();
    zerollm_backend::check_last_error("TransformerDecoder::backward() failed");
    LOG_DEBUG("TransformerDecoder backward completed");
}

/**
 * @brief 保存模型参数
 * 
 * @param path 保存路径
 */
void TransformerDecoder::save(const std::string& path) {
    LOG_DEBUG("Saving TransformerDecoder to " << path);
    
    // 创建目录
    Serializer::create_directory(path);
    
    // 保存每一层
    for (int i = 0; i < num_layers_; ++i) {
        std::string layer_path = path + "/layer_" + std::to_string(i);
        layers_[i]->save(layer_path);
    }
    
    LOG_DEBUG("TransformerDecoder saved successfully");
}

/**
 * @brief 加载模型参数
 * 
 * @param path 加载路径
 */
void TransformerDecoder::load(const std::string& path) {
    LOG_DEBUG("Loading TransformerDecoder from " << path);
    
    // 加载每一层
    for (int i = 0; i < num_layers_; ++i) {
        std::string layer_path = path + "/layer_" + std::to_string(i);
        layers_[i]->load(layer_path);
    }
    
    LOG_DEBUG("TransformerDecoder loaded successfully");
}

/**
 * @brief 设置优化器
 * 
 * @param config 优化器配置
 */
void TransformerDecoder::set_optimizer(OptimizerConfig config) {
    LOG_DEBUG("Setting optimizer for TransformerDecoder with " << layers_.size() << " layers");
    for (auto layer : layers_) {
        layer->set_optimizer(config);
    }
    LOG_DEBUG("TransformerDecoder optimizer set");
}

/**
 * @brief 优化器步进
 * 
 * @param learning_rate 学习率
 */
void TransformerDecoder::step(float learning_rate) {
    LOG_DEBUG("TransformerDecoder optimizer step with learning_rate=" << learning_rate);
    for (auto layer : layers_) {
        layer->step(learning_rate);
    }
    LOG_DEBUG("TransformerDecoder optimizer step completed");
}

/**
 * @brief 清零所有梯度
 */
void TransformerDecoder::zero_grad() {
    LOG_DEBUG("TransformerDecoder zeroing gradients");
    for (auto layer : layers_) {
        layer->zero_grad();
    }
    LOG_DEBUG("TransformerDecoder gradients zeroed");
}