#include "zerollm.hpp"
#include <cmath>
#include <iostream>
#include "async_logger.hpp"
#include "serializer.hpp"

ZeroLLM::ZeroLLM(const ZeroLLMConfig& config)
    : config_(config),
      embedding_(nullptr),
      decoder_(nullptr),
      output_ln_(nullptr),
      hidden_states_(nullptr),
      logits_(nullptr),
      last_batch_size_(0),
      last_seq_len_(0) {
    
    LOG_INFO("Initializing ZeroLLM model with config: layers=" << config.num_layers 
             << ", embed_dim=" << config.embed_dim << ", num_heads=" << config.num_heads
             << ", ff_hidden_dim=" << config.ff_hidden_dim << ", vocab_size=" << config.vocab_size);
    
    // 创建嵌入层
    embedding_ = new Embedding(
        config_.vocab_size,
        config_.embed_dim,
        config_.max_seq_len,
        config_.with_grad
    );
    
    // 创建Transformer解码器
    decoder_ = new TransformerDecoder(
        config_.num_layers,
        config_.embed_dim,
        config_.num_heads,
        config_.ff_hidden_dim,
        config_.with_grad
    );
    
    // 创建输出层归一化（可选，通常用于稳定训练）
    output_ln_ = new LayerNorm(
        config_.embed_dim,
        config_.with_grad,
        1e-5f
    );
    
    LOG_INFO("ZeroLLM model initialized successfully with " << calculate_num_params() << "M parameters");
}



ZeroLLM::~ZeroLLM() {
    delete embedding_;
    delete decoder_;
    delete output_ln_;
    
    zerollm_backend::free(hidden_states_);
    zerollm_backend::free(logits_);
}

double ZeroLLM::calculate_num_params() const {
    double num_params = 0.0;
    
    // Embedding层参数: vocab_size * embed_dim
    num_params += (double)config_.vocab_size * config_.embed_dim;
    
    // Transformer Decoder层参数
    // 每层包含：
    // 1. Multi-Head Attention: 4个Linear层 (Q, K, V, O)
    //    - 每个: embed_dim * embed_dim + embed_dim (bias)
    // 2. Feed-Forward Network: 2个Linear层
    //    - FF1: embed_dim * ff_hidden_dim + ff_hidden_dim
    //    - FF2: ff_hidden_dim * embed_dim + embed_dim
    // 3. LayerNorm: 2个 (每个有 gamma 和 beta)
    //    - 每个: 2 * embed_dim
    
    double params_per_layer = 0.0;
    
    // MHA参数
    params_per_layer += 4.0 * (config_.embed_dim * config_.embed_dim + config_.embed_dim);
    
    // FFN参数
    params_per_layer += config_.embed_dim * config_.ff_hidden_dim + config_.ff_hidden_dim;
    params_per_layer += config_.ff_hidden_dim * config_.embed_dim + config_.embed_dim;
    
    // LayerNorm参数 (2个)
    params_per_layer += 2.0 * 2.0 * config_.embed_dim;
    
    // 总层数参数
    num_params += params_per_layer * config_.num_layers;
    
    // Output LayerNorm参数
    num_params += 2.0 * config_.embed_dim;
    
    // 注意：Output层与Embedding共享权重，不额外计算
    
    return num_params / 1e6; // 转换为百万
}

void ZeroLLM::forward(float* logits, const int* input_ids, int batch_size, int seq_len) {
    LOG_DEBUG("ZeroLLM forward pass started: batch_size=" << batch_size << ", seq_len=" << seq_len);
    
    // 检查序列长度
    if (seq_len > config_.max_seq_len) {
        LOG_ERROR("Sequence length " << seq_len << " exceeds max_seq_len " << config_.max_seq_len);
        throw std::runtime_error("Sequence length exceeds max_seq_len");
    }
    
    // 分配或调整缓冲区大小
    if (last_batch_size_ != batch_size || last_seq_len_ != seq_len) {
        LOG_DEBUG("Resizing buffers for new batch/seq size: " << batch_size << "x" << seq_len);
        zerollm_backend::free(hidden_states_);
        zerollm_backend::free(logits_);
        
        int hidden_size = batch_size * seq_len * config_.embed_dim;
        int logits_size = batch_size * seq_len * config_.vocab_size;
        
        hidden_states_ = (float*)zerollm_backend::malloc(hidden_size * sizeof(float));
        logits_ = (float*)zerollm_backend::malloc(logits_size * sizeof(float));
        
        last_batch_size_ = batch_size;
        last_seq_len_ = seq_len;
    }
    
    // 1. Embedding层: input_ids -> hidden_states
    LOG_DEBUG("Forward through embedding layer");
    embedding_->forward(hidden_states_, input_ids, batch_size, seq_len);
    
    // 2. Transformer Decoder: hidden_states -> hidden_states
    LOG_DEBUG("Forward through transformer decoder with " << decoder_->num_layers() << " layers");
    decoder_->forward(hidden_states_, hidden_states_, batch_size, seq_len);
    
    // 3. Output LayerNorm (可选，用于稳定训练)
    LOG_DEBUG("Forward through output layer norm");
    float* ln_output = hidden_states_; // 可以in-place操作
    output_ln_->forward(ln_output, hidden_states_, batch_size * seq_len);
    
    // 4. Output层: hidden_states @ embedding_weight^T -> logits
    // embedding权重形状: [vocab_size, embed_dim]
    // 我们需要计算: logits = hidden_states @ embedding_weight^T
    // hidden_states: [batch_size * seq_len, embed_dim]
    // embedding_weight: [vocab_size, embed_dim]
    // embedding_weight^T: [embed_dim, vocab_size]
    // logits: [batch_size * seq_len, vocab_size]
    // 
    // 注意: matmul(A, B) 计算 A @ B，其中 A是[M, K]，B是[K, N]
    // 我们需要: hidden_states [M, K] @ embedding_weight^T [K, N]
    // 所以需要转置embedding_weight
    const float* embedding_weight = embedding_->embedding_table_device();
    
    // 分配临时缓冲区用于转置
    LOG_DEBUG("Allocating temporary buffer for embedding weight transpose");
    float* embedding_weight_T = (float*)zerollm_backend::malloc(
        config_.vocab_size * config_.embed_dim * sizeof(float)
    );
    
    // 转置embedding权重: [vocab_size, embed_dim] -> [embed_dim, vocab_size]
    LOG_DEBUG("Transposing embedding weights");
    transpose<float>(
        embedding_weight,
        embedding_weight_T,
        config_.vocab_size,      // 原矩阵行数
        config_.embed_dim,        // 原矩阵列数
        0                         // stream
    );
    
    // 计算 logits = hidden_states @ embedding_weight^T
    LOG_DEBUG("Computing logits with matmul");
    matmul_tiled<float>(
        hidden_states_,           // [batch_size * seq_len, embed_dim] = [M, K]
        embedding_weight_T,       // [embed_dim, vocab_size] = [K, N]
        logits_,                  // [batch_size * seq_len, vocab_size] = [M, N]
        batch_size * seq_len,     // M
        config_.vocab_size,       // N
        config_.embed_dim,        // K
        0                         // stream
    );
    
    zerollm_backend::free(embedding_weight_T);
    
    // 将logits复制到输出
    LOG_DEBUG("Copying logits to output buffer");
    int logits_size = batch_size * seq_len * config_.vocab_size;
    zerollm_backend::memcpy(
        logits,
        logits_,
        logits_size * sizeof(float),
        zerollm_backend::CopyKind::D2D
    );
    
    zerollm_backend::device_synchronize();
    zerollm_backend::check_last_error("ZeroLLM::forward() failed");
    LOG_DEBUG("ZeroLLM forward pass completed");
}

void ZeroLLM::backward(const int* input_ids, const float* d_logits, int batch_size, int seq_len) {
    LOG_DEBUG("ZeroLLM backward pass started: batch_size=" << batch_size << ", seq_len=" << seq_len);
    
    if (!config_.with_grad) {
        LOG_ERROR("Cannot call backward() when with_grad=false");
        throw std::runtime_error("Cannot call backward() when with_grad=false");
    }
    
    if (last_batch_size_ != batch_size || last_seq_len_ != seq_len) {
        LOG_ERROR("Must call forward() with same batch_size and seq_len before backward()");
        throw std::runtime_error("Must call forward() with same batch_size and seq_len before backward()");
    }
    
    int hidden_size = batch_size * seq_len * config_.embed_dim;
    int logits_size = batch_size * seq_len * config_.vocab_size;
    
    // 分配临时梯度缓冲区
    LOG_DEBUG("Allocating temporary gradient buffers");
    float* d_hidden_states = (float*)zerollm_backend::malloc(hidden_size * sizeof(float));
    float* d_logits_temp = (float*)zerollm_backend::malloc(logits_size * sizeof(float));
    float* d_ln_output = (float*)zerollm_backend::malloc(hidden_size * sizeof(float));
    
    // 复制d_logits到临时缓冲区
    LOG_DEBUG("Copying d_logits to temporary buffer");
    zerollm_backend::memcpy(
        d_logits_temp,
        d_logits,
        logits_size * sizeof(float),
        zerollm_backend::CopyKind::D2D
    );
    
    // 1. Backward through output layer: d_logits -> d_hidden_states
    // 前向: logits = hidden_states @ embedding_weight^T
    // 反向: d_hidden_states = d_logits @ embedding_weight
    // d_logits: [batch_size * seq_len, vocab_size] = [M, N]
    // embedding_weight: [vocab_size, embed_dim] = [N, K]
    // d_hidden_states: [batch_size * seq_len, embed_dim] = [M, K]
    const float* embedding_weight = embedding_->embedding_table_device();
    
    LOG_DEBUG("Backward through output layer");
    matmul_tiled<float>(
        d_logits_temp,            // [batch_size * seq_len, vocab_size] = [M, N]
        embedding_weight,         // [vocab_size, embed_dim] = [N, K]
        d_hidden_states,          // [batch_size * seq_len, embed_dim] = [M, K]
        batch_size * seq_len,     // M
        config_.embed_dim,        // K
        config_.vocab_size,       // N
        0                         // stream
    );
    
    // 2. Backward through embedding weight (需要累加到embedding的梯度)
    // 前向: logits = hidden_states @ embedding_weight^T
    // 反向: d_embedding_weight = d_logits^T @ hidden_states
    // d_logits: [batch_size * seq_len, vocab_size] = [M, N]
    // hidden_states: [batch_size * seq_len, embed_dim] = [M, K]
    // d_embedding_weight: [vocab_size, embed_dim] = [N, K]
    // 计算: d_embedding_weight^T = hidden_states^T @ d_logits
    // 即: d_embedding_weight = (hidden_states^T @ d_logits)^T = d_logits^T @ hidden_states
    float* d_embedding_weight = embedding_->d_embedding_table();
    if (d_embedding_weight) {
        // 分配临时缓冲区
        LOG_DEBUG("Computing embedding weight gradients");
        float* d_embedding_weight_T = (float*)zerollm_backend::malloc(
            config_.embed_dim * config_.vocab_size * sizeof(float)
        );
        
        // 计算 d_embedding_weight^T = hidden_states^T @ d_logits
        // hidden_states^T: [embed_dim, batch_size * seq_len] = [K, M]
        // d_logits: [batch_size * seq_len, vocab_size] = [M, N]
        // d_embedding_weight^T: [embed_dim, vocab_size] = [K, N]
        matmul_tiled<float>(
            hidden_states_,       // [batch_size * seq_len, embed_dim] = [M, K]
            d_logits_temp,        // [batch_size * seq_len, vocab_size] = [M, N]
            d_embedding_weight_T, // [embed_dim, vocab_size] = [K, N] (转置后的结果)
            config_.embed_dim,    // K
            config_.vocab_size,   // N
            batch_size * seq_len, // M
            0                     // stream
        );
        
        // 转置回原始形状
        // d_embedding_weight_T: [embed_dim, vocab_size]
        // d_embedding_weight: [vocab_size, embed_dim]
        // 注意：我们需要累加梯度，所以先转置到临时缓冲区，然后累加
        float* d_embedding_weight_temp = (float*)zerollm_backend::malloc(
            config_.vocab_size * config_.embed_dim * sizeof(float)
        );
        
        transpose<float>(
            d_embedding_weight_T,
            d_embedding_weight_temp,
            config_.embed_dim,    // 原矩阵行数
            config_.vocab_size,   // 原矩阵列数
            0                     // stream
        );
        
        // 累加梯度到embedding的梯度缓冲区
        // 注意：embedding的backward可能也会计算梯度（从input_ids），
        // 但通常output层的梯度是主要的，所以这里直接累加
        int embedding_grad_size = config_.vocab_size * config_.embed_dim;
        add_inplace<float>(
            d_embedding_weight,
            d_embedding_weight_temp,
            embedding_grad_size,
            0
        );
        
        zerollm_backend::free(d_embedding_weight_T);
        zerollm_backend::free(d_embedding_weight_temp);
    }
    
    // 3. Backward through output LayerNorm
    LOG_DEBUG("Backward through output layer norm");
    output_ln_->backward(d_ln_output, d_hidden_states);
    
    // 4. Backward through Transformer Decoder
    LOG_DEBUG("Backward through transformer decoder");
    decoder_->backward(d_ln_output, d_ln_output);
    
    // 5. Backward through Embedding
    // 注意：embedding的backward通常是从input_ids计算的（用于更新权重），
    // 但这里我们已经从output层计算了embedding权重的梯度并累加了。
    // 如果embedding层需要从hidden_states反向传播（虽然通常不需要），
    // 可以在这里调用embedding_->backward，但需要传入d_ln_output作为d_output
    // 由于input_ids是离散的，通常不需要计算input_ids的梯度，
    // 所以这里我们跳过embedding的backward（权重梯度已经在步骤2中处理了）
    
    // 清理临时缓冲区
    LOG_DEBUG("Cleaning up temporary buffers");
    zerollm_backend::free(d_hidden_states);
    zerollm_backend::free(d_logits_temp);
    zerollm_backend::free(d_ln_output);
    
    zerollm_backend::device_synchronize();
    zerollm_backend::check_last_error("ZeroLLM::backward() failed");
    LOG_DEBUG("ZeroLLM backward pass completed");
}


/**
 * @brief 保存模型参数
 * * 保存策略：
 * 1. 创建主目录
 * 2. 保存 Config 配置信息（二进制dump，用于校验）
 * 3. 递归调用各子模块的 save 方法
 * * @param path 保存路径 (文件夹路径)
 */
void ZeroLLM::save(const std::string& path) {
    LOG_INFO("Saving ZeroLLM model to " << path);
    
    // 1. 创建目录
    Serializer::create_directory(path);
    
    // 2. 保存配置信息 (Config)
    // 将 config_ 结构体直接写入二进制文件，方便下次加载时校验参数是否匹配
    std::string config_path = path + "/config.bin";
    std::ofstream config_file(config_path, std::ios::binary);
    if (config_file.is_open()) {
        config_file.write(reinterpret_cast<const char*>(&config_), sizeof(ZeroLLMConfig));
        config_file.close();
        LOG_DEBUG("Config saved to " << config_path);
    } else {
        LOG_ERROR("Failed to open file for writing config: " << config_path);
        // 这里可以选择抛出异常，或者仅记录错误
    }

    // 3. 保存 Embedding 层
    if (embedding_) {
        embedding_->save(path + "/embedding");
    }

    // 4. 保存 Decoder 层 (包含所有 Transformer Blocks)
    if (decoder_) {
        decoder_->save(path + "/decoder");
    }

    // 5. 保存 Output LayerNorm
    if (output_ln_) {
        output_ln_->save(path + "/output_ln");
    }
    
    LOG_INFO("ZeroLLM model saved successfully");
}

/**
 * @brief 加载模型参数
 * * 加载策略：
 * 1. (可选) 读取 config.bin 校验当前模型结构是否与文件匹配
 * 2. 递归调用各子模块的 load 方法
 * * @param path 加载路径 (文件夹路径)
 */
void ZeroLLM::load(const std::string& path) {
    LOG_INFO("Loading ZeroLLM model from " << path);
    
    // 1. 简单的配置校验 (可选)
    std::string config_path = path + "/config.bin";
    std::ifstream config_file(config_path, std::ios::binary);
    if (config_file.is_open()) {
        ZeroLLMConfig file_config(0,0,0,0,0,0,false);
        config_file.read(reinterpret_cast<char*>(&file_config), sizeof(ZeroLLMConfig));
        config_file.close();

        // 简单的完整性检查：比如检查维度是否一致
        if (file_config.embed_dim != config_.embed_dim || 
            file_config.num_layers != config_.num_layers || 
            file_config.vocab_size != config_.vocab_size) {
            
            LOG_ERROR("Config mismatch! Model initialized with embed_dim=" << config_.embed_dim 
                      << " but file has " << file_config.embed_dim);
            throw std::runtime_error("Model configuration mismatch detected during load.");
        }
        LOG_DEBUG("Config verification passed.");
    } else {
        LOG_WARN("No config file found at " << config_path << ", skipping verification.");
    }
    
    // 2. 加载 Embedding 层
    if (embedding_) {
        embedding_->load(path + "/embedding");
    } else {
        throw std::runtime_error("Embedding layer is null during load");
    }

    // 3. 加载 Decoder 层
    if (decoder_) {
        decoder_->load(path + "/decoder");
    } else {
        throw std::runtime_error("Decoder layer is null during load");
    }

    // 4. 加载 Output LayerNorm
    if (output_ln_) {
        output_ln_->load(path + "/output_ln");
    } else {
        throw std::runtime_error("Output LayerNorm is null during load");
    }
    
    LOG_INFO("ZeroLLM model loaded successfully");
}

void ZeroLLM::zero_grad() {
    LOG_DEBUG("ZeroLLM zeroing gradients");
    
    if (!config_.with_grad) {
        LOG_DEBUG("Gradient computation disabled, skipping zero_grad");
        return;
    }
    
    embedding_->zero_grad();
    decoder_->zero_grad();
    output_ln_->zero_grad();
    
    LOG_DEBUG("ZeroLLM gradients zeroed");
}

void ZeroLLM::set_optimizer(OptimizerConfig config) {
    LOG_INFO("Setting optimizer for ZeroLLM model");
    embedding_->set_optimizer(config);
    decoder_->set_optimizer(config);
    output_ln_->set_optimizer(config);
    LOG_DEBUG("ZeroLLM optimizer set");
}

void ZeroLLM::step(float learning_rate) {
    LOG_DEBUG("ZeroLLM optimizer step with learning_rate=" << learning_rate);
    
    if (!config_.with_grad) {
        LOG_ERROR("Cannot call step() when with_grad=false");
        throw std::runtime_error("Cannot call step() when with_grad=false");
    }
    
    embedding_->step(learning_rate);
    decoder_->step(learning_rate);
    output_ln_->step(learning_rate);
    
    LOG_DEBUG("ZeroLLM optimizer step completed");
}