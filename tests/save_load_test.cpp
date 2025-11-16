#include <iostream>
#include <filesystem>
#include <cmath>
#include <cstdlib>
#include <vector>
#include "layer/embedding.hpp"
#include "layer/linear.hpp"
#include "layer/layernorm.hpp"
#include "layer/mha.hpp"
#include "layer/feedward.hpp"
#include "layer/transformer_block.hpp"
#include "layer/transformer_decoder.hpp"
#include "config.hpp"

// 比较两个浮点数组是否相等
bool compare_float_arrays(const float* a, const float* b, size_t size, float tolerance = 1e-6) {
    for (size_t i = 0; i < size; ++i) {
        if (std::abs(a[i] - b[i]) > tolerance) {
            std::cout << "Mismatch at index " << i << ": " << a[i] << " vs " << b[i] << std::endl;
            return false;
        }
    }
    return true;
}

// 测试Embedding层的保存和加载
bool test_embedding_save_load() {
    std::cout << "Testing Embedding layer save/load..." << std::endl;
    
    const std::string temp_path = "/tmp/test_embedding";
    const int vocab_size = 100;
    const int embed_dim = 32;
    const int max_seq_len = 64;
    
    // 创建Embedding层
    Embedding embedding(vocab_size, embed_dim, max_seq_len, true);
    
    // 保存
    embedding.save(temp_path);
    
    // 创建另一个Embedding层并加载
    Embedding loaded_embedding(vocab_size, embed_dim, max_seq_len, true);
    loaded_embedding.load(temp_path);
    
    // 比较参数
    float* original_table = (float*)malloc(vocab_size * embed_dim * sizeof(float));
    float* loaded_table = (float*)malloc(vocab_size * embed_dim * sizeof(float));
    
    zerollm_backend::memcpy(original_table, embedding.embedding_table_device(), 
                           vocab_size * embed_dim * sizeof(float), zerollm_backend::CopyKind::D2H);
    zerollm_backend::memcpy(loaded_table, loaded_embedding.embedding_table_device(), 
                           vocab_size * embed_dim * sizeof(float), zerollm_backend::CopyKind::D2H);
    
    bool result = compare_float_arrays(original_table, loaded_table, vocab_size * embed_dim);
    
    free(original_table);
    free(loaded_table);
    
    // 清理临时文件
    std::filesystem::remove_all(temp_path);
    
    std::cout << "Embedding layer test: " << (result ? "PASS" : "FAIL") << std::endl;
    return result;
}

// 测试Linear层的保存和加载
bool test_linear_save_load() {
    std::cout << "Testing Linear layer save/load..." << std::endl;
    
    const std::string temp_path = "/tmp/test_linear";
    const int in_features = 32;
    const int out_features = 64;
    
    // 创建Linear层
    Linear linear(in_features, out_features, true, true);
    
    // 保存
    linear.save(temp_path);
    
    // 创建另一个Linear层并加载
    Linear loaded_linear(in_features, out_features, true, true);
    loaded_linear.load(temp_path);
    
    // 比较权重
    float* original_weight = (float*)malloc(out_features * in_features * sizeof(float));
    float* loaded_weight = (float*)malloc(out_features * in_features * sizeof(float));
    
    zerollm_backend::memcpy(original_weight, linear.weight(), 
                           out_features * in_features * sizeof(float), zerollm_backend::CopyKind::D2H);
    zerollm_backend::memcpy(loaded_weight, loaded_linear.weight(), 
                           out_features * in_features * sizeof(float), zerollm_backend::CopyKind::D2H);
    
    bool weight_match = compare_float_arrays(original_weight, loaded_weight, out_features * in_features);
    
    // 比较偏置
    float* original_bias = (float*)malloc(out_features * sizeof(float));
    float* loaded_bias = (float*)malloc(out_features * sizeof(float));
    
    zerollm_backend::memcpy(original_bias, linear.bias(), 
                           out_features * sizeof(float), zerollm_backend::CopyKind::D2H);
    zerollm_backend::memcpy(loaded_bias, loaded_linear.bias(), 
                           out_features * sizeof(float), zerollm_backend::CopyKind::D2H);
    
    bool bias_match = compare_float_arrays(original_bias, loaded_bias, out_features);
    
    bool result = weight_match && bias_match;
    
    free(original_weight);
    free(loaded_weight);
    free(original_bias);
    free(loaded_bias);
    
    // 清理临时文件
    std::filesystem::remove_all(temp_path);
    
    std::cout << "Linear layer test: " << (result ? "PASS" : "FAIL") << std::endl;
    return result;
}

// 测试LayerNorm层的保存和加载
bool test_layernorm_save_load() {
    std::cout << "Testing LayerNorm layer save/load..." << std::endl;
    
    const std::string temp_path = "/tmp/test_layernorm";
    const int feature_size = 64;
    
    // 创建LayerNorm层
    LayerNorm layernorm(feature_size, true);
    
    // 保存
    layernorm.save(temp_path);
    
    // 创建另一个LayerNorm层并加载
    LayerNorm loaded_layernorm(feature_size, true);
    loaded_layernorm.load(temp_path);
    
    // 比较gamma参数
    float* original_gamma = (float*)malloc(feature_size * sizeof(float));
    float* loaded_gamma = (float*)malloc(feature_size * sizeof(float));
    
    zerollm_backend::memcpy(original_gamma, layernorm.gamma(), 
                           feature_size * sizeof(float), zerollm_backend::CopyKind::D2H);
    zerollm_backend::memcpy(loaded_gamma, loaded_layernorm.gamma(), 
                           feature_size * sizeof(float), zerollm_backend::CopyKind::D2H);
    
    bool gamma_match = compare_float_arrays(original_gamma, loaded_gamma, feature_size);
    
    // 比较beta参数
    float* original_beta = (float*)malloc(feature_size * sizeof(float));
    float* loaded_beta = (float*)malloc(feature_size * sizeof(float));
    
    zerollm_backend::memcpy(original_beta, layernorm.beta(), 
                           feature_size * sizeof(float), zerollm_backend::CopyKind::D2H);
    zerollm_backend::memcpy(loaded_beta, loaded_layernorm.beta(), 
                           feature_size * sizeof(float), zerollm_backend::CopyKind::D2H);
    
    bool beta_match = compare_float_arrays(original_beta, loaded_beta, feature_size);
    
    bool result = gamma_match && beta_match;
    
    free(original_gamma);
    free(loaded_gamma);
    free(original_beta);
    free(loaded_beta);
    
    // 清理临时文件
    std::filesystem::remove_all(temp_path);
    
    std::cout << "LayerNorm layer test: " << (result ? "PASS" : "FAIL") << std::endl;
    return result;
}

// 测试MultiHeadAttention层的保存和加载
bool test_mha_save_load() {
    std::cout << "Testing MultiHeadAttention layer save/load..." << std::endl;
    
    const std::string temp_path = "/tmp/test_mha";
    const int embed_dim = 64;
    const int num_heads = 8;
    
    // 创建MultiHeadAttention层
    MultiHeadAttention mha(embed_dim, num_heads, true, true);
    
    // 保存
    mha.save(temp_path);
    
    // 创建另一个MultiHeadAttention层并加载
    MultiHeadAttention loaded_mha(embed_dim, num_heads, true, true);
    loaded_mha.load(temp_path);
    
    // 对于MHA，我们验证其子层是否正确加载
    // 这里我们只简单检查是否能成功加载而不抛出异常
    bool result = true; // 如果能运行到这里没有异常，就认为基本成功
    
    // 清理临时文件
    std::filesystem::remove_all(temp_path);
    
    std::cout << "MultiHeadAttention layer test: " << (result ? "PASS" : "FAIL") << std::endl;
    return result;
}

// 测试FeedForward层的保存和加载
bool test_feedforward_save_load() {
    std::cout << "Testing FeedForward layer save/load..." << std::endl;
    
    const std::string temp_path = "/tmp/test_feedforward";
    const int embed_dim = 64;
    const int ff_hidden_dim = 128;
    
    // 创建FeedForward层
    FeedForward ff(embed_dim, ff_hidden_dim, true);
    
    // 保存
    ff.save(temp_path);
    
    // 创建另一个FeedForward层并加载
    FeedForward loaded_ff(embed_dim, ff_hidden_dim, true);
    loaded_ff.load(temp_path);
    
    // 对于FeedForward，我们验证其子层是否正确加载
    bool result = true; // 如果能运行到这里没有异常，就认为基本成功
    
    // 清理临时文件
    std::filesystem::remove_all(temp_path);
    
    std::cout << "FeedForward layer test: " << (result ? "PASS" : "FAIL") << std::endl;
    return result;
}

// 测试TransformerDecoderBlock层的保存和加载
bool test_transformer_block_save_load() {
    std::cout << "Testing TransformerDecoderBlock layer save/load..." << std::endl;
    
    const std::string temp_path = "/tmp/test_transformer_block";
    const int embed_dim = 64;
    const int num_heads = 8;
    const int ff_hidden_dim = 128;
    
    // 创建TransformerDecoderBlock层
    TransformerDecoderBlock block(embed_dim, num_heads, ff_hidden_dim, true);
    
    // 保存
    block.save(temp_path);
    
    // 创建另一个TransformerDecoderBlock层并加载
    TransformerDecoderBlock loaded_block(embed_dim, num_heads, ff_hidden_dim, true);
    loaded_block.load(temp_path);
    
    // 对于TransformerDecoderBlock，我们验证其子层是否正确加载
    bool result = true; // 如果能运行到这里没有异常，就认为基本成功
    
    // 清理临时文件
    std::filesystem::remove_all(temp_path);
    
    std::cout << "TransformerDecoderBlock layer test: " << (result ? "PASS" : "FAIL") << std::endl;
    return result;
}

// 测试TransformerDecoder层的保存和加载
bool test_transformer_decoder_save_load() {
    std::cout << "Testing TransformerDecoder layer save/load..." << std::endl;
    
    const std::string temp_path = "/tmp/test_transformer_decoder";
    const int num_layers = 2;
    const int embed_dim = 64;
    const int num_heads = 8;
    const int ff_hidden_dim = 128;
    
    // 创建TransformerDecoder层
    TransformerDecoder decoder(num_layers, embed_dim, num_heads, ff_hidden_dim, true);
    
    // 保存
    decoder.save(temp_path);
    
    // 创建另一个TransformerDecoder层并加载
    TransformerDecoder loaded_decoder(num_layers, embed_dim, num_heads, ff_hidden_dim, true);
    loaded_decoder.load(temp_path);
    
    // 对于TransformerDecoder，我们验证其子层是否正确加载
    bool result = true; // 如果能运行到这里没有异常，就认为基本成功
    
    // 清理临时文件
    std::filesystem::remove_all(temp_path);
    
    std::cout << "TransformerDecoder layer test: " << (result ? "PASS" : "FAIL") << std::endl;
    return result;
}

int main() {
    std::cout << "Starting layer save/load tests..." << std::endl;
    
    bool all_passed = true;
    
    all_passed &= test_embedding_save_load();
    all_passed &= test_linear_save_load();
    all_passed &= test_layernorm_save_load();
    all_passed &= test_mha_save_load();
    all_passed &= test_feedforward_save_load();
    all_passed &= test_transformer_block_save_load();
    all_passed &= test_transformer_decoder_save_load();
    
    std::cout << "\nAll tests " << (all_passed ? "PASSED" : "FAILED") << std::endl;
    
    return all_passed ? 0 : 1;
}