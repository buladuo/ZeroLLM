#include <iostream>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <vector>

// 包含 CUDA 运行时
#include <cuda_runtime.h>

// 包含我们要测试的 mha kernel 函数
#include "kernel/cuda/mha.cuh"
#include "config.hpp"

#define EPSILON 1e-3f

bool compare_float(float a, float b) {
    return fabsf(a - b) < EPSILON;
}

void init_matrix(float* matrix, int size, float factor = 1.0f) {
    for (int i = 0; i < size; i++) {
        matrix[i] = (static_cast<float>(rand()) / RAND_MAX) * 2.0f * factor - factor;
    }
}

void test_mha_forward() {
    std::cout << "Running MHA forward test..." << std::endl;
    
    const int64_t batch_size = 2;
    const int64_t seq_len = 4;
    const int64_t num_heads = 2;
    const int64_t head_dim = 3;
    const int64_t embed_dim = num_heads * head_dim; // 6
    
    // 输入张量 [batch_size, seq_len, embed_dim]
    float *h_Q = (float*)malloc(batch_size * seq_len * embed_dim * sizeof(float));
    float *h_K = (float*)malloc(batch_size * seq_len * embed_dim * sizeof(float));
    float *h_V = (float*)malloc(batch_size * seq_len * embed_dim * sizeof(float));
    
    // 输出张量
    float *h_attn_output = (float*)malloc(batch_size * seq_len * embed_dim * sizeof(float));
    float *h_attention_scores = (float*)malloc(batch_size * num_heads * seq_len * seq_len * sizeof(float));
    
    // 初始化输入数据
    srand(12345); // 固定种子以确保结果可重现
    init_matrix(h_Q, batch_size * seq_len * embed_dim, 1.0f);
    init_matrix(h_K, batch_size * seq_len * embed_dim, 1.0f);
    init_matrix(h_V, batch_size * seq_len * embed_dim, 1.0f);
    
    // 清零输出
    for (int i = 0; i < batch_size * seq_len * embed_dim; i++) {
        h_attn_output[i] = 0.0f;
    }
    for (int i = 0; i < batch_size * num_heads * seq_len * seq_len; i++) {
        h_attention_scores[i] = 0.0f;
    }

    // 分配设备内存
    float *d_Q, *d_K, *d_V, *d_attn_output, *d_attention_scores;
    d_Q = (float *)zerollm_backend::malloc(batch_size * seq_len * embed_dim * sizeof(float));
    d_K = (float *)zerollm_backend::malloc(batch_size * seq_len * embed_dim * sizeof(float));
    d_V = (float *)zerollm_backend::malloc(batch_size * seq_len * embed_dim * sizeof(float));
    d_attn_output = (float *)zerollm_backend::malloc(batch_size * seq_len * embed_dim * sizeof(float));
    d_attention_scores = (float *)zerollm_backend::malloc(batch_size * num_heads * seq_len * seq_len * sizeof(float));

    // 将数据从主机复制到设备
    zerollm_backend::memcpy(d_Q, h_Q, batch_size * seq_len * embed_dim * sizeof(float), zerollm_backend::CopyKind::H2D);
    zerollm_backend::memcpy(d_K, h_K, batch_size * seq_len * embed_dim * sizeof(float), zerollm_backend::CopyKind::H2D);
    zerollm_backend::memcpy(d_V, h_V, batch_size * seq_len * embed_dim * sizeof(float), zerollm_backend::CopyKind::H2D);
    zerollm_backend::memcpy(d_attn_output, h_attn_output, batch_size * seq_len * embed_dim * sizeof(float), zerollm_backend::CopyKind::H2D);
    zerollm_backend::memcpy(d_attention_scores, h_attention_scores, batch_size * num_heads * seq_len * seq_len * sizeof(float), zerollm_backend::CopyKind::H2D);

    // 执行 MHA forward 操作 (非因果模式)
    mha_forward<float>(d_Q, d_K, d_V, d_attn_output, d_attention_scores,
                       nullptr, // mask
                       batch_size, seq_len, num_heads, head_dim,
                       false, // is_causal
                       0);

    // 将结果从设备复制回主机
    zerollm_backend::memcpy(h_attn_output, d_attn_output, batch_size * seq_len * embed_dim * sizeof(float), zerollm_backend::CopyKind::D2H);
    zerollm_backend::memcpy(h_attention_scores, d_attention_scores, batch_size * num_heads * seq_len * seq_len * sizeof(float), zerollm_backend::CopyKind::D2H);

    // 验证输出不为零（基本功能检查）
    bool success = true;
    bool found_nonzero_output = false;
    bool found_nonzero_scores = false;
    
    for (int i = 0; i < batch_size * seq_len * embed_dim; i++) {
        if (fabsf(h_attn_output[i]) > 1e-6) {
            found_nonzero_output = true;
            break;
        }
    }
    
    for (int i = 0; i < batch_size * num_heads * seq_len * seq_len; i++) {
        if (fabsf(h_attention_scores[i]) > 1e-6) {
            found_nonzero_scores = true;
            break;
        }
    }

    if (!found_nonzero_output) {
        success = false;
        std::cerr << "MHA forward output is all zeros!" << std::endl;
    }
    
    if (!found_nonzero_scores) {
        success = false;
        std::cerr << "MHA attention scores are all zeros!" << std::endl;
    }

    if (success) {
        std::cout << "MHA forward test passed!" << std::endl;
    } else {
        std::cerr << "MHA forward test failed!" << std::endl;
    }

    // 释放内存
    zerollm_backend::free(d_Q);
    zerollm_backend::free(d_K);
    zerollm_backend::free(d_V);
    zerollm_backend::free(d_attn_output);
    zerollm_backend::free(d_attention_scores);
    free(h_Q);
    free(h_K);
    free(h_V);
    free(h_attn_output);
    free(h_attention_scores);
}

void test_mha_forward_causal() {
    std::cout << "Running MHA forward test with causal mask..." << std::endl;
    
    const int64_t batch_size = 1;
    const int64_t seq_len = 3;
    const int64_t num_heads = 1;
    const int64_t head_dim = 2;
    const int64_t embed_dim = num_heads * head_dim; // 2
    
    // 输入张量 [batch_size, seq_len, embed_dim]
    float *h_Q = (float*)malloc(batch_size * seq_len * embed_dim * sizeof(float));
    float *h_K = (float*)malloc(batch_size * seq_len * embed_dim * sizeof(float));
    float *h_V = (float*)malloc(batch_size * seq_len * embed_dim * sizeof(float));
    
    // 输出张量
    float *h_attn_output = (float*)malloc(batch_size * seq_len * embed_dim * sizeof(float));
    float *h_attention_scores = (float*)malloc(batch_size * num_heads * seq_len * seq_len * sizeof(float));
    
    // 初始化简单的输入数据以便验证因果掩码
    // Q = [1, 0, 0, 1, 1, 1] => [[[1, 0], [0, 1], [1, 1]]]
    // K = [1, 0, 0, 1, 1, 1] => [[[1, 0], [0, 1], [1, 1]]]
    // V = [1, 2, 3, 4, 5, 6] => [[[1, 2], [3, 4], [5, 6]]]
    float Q_vals[] = {1, 0, 0, 1, 1, 1};
    float K_vals[] = {1, 0, 0, 1, 1, 1};
    float V_vals[] = {1, 2, 3, 4, 5, 6};
    
    for (int i = 0; i < batch_size * seq_len * embed_dim; i++) {
        h_Q[i] = Q_vals[i];
        h_K[i] = K_vals[i];
        h_V[i] = V_vals[i];
    }
    
    // 清零输出
    for (int i = 0; i < batch_size * seq_len * embed_dim; i++) {
        h_attn_output[i] = 0.0f;
    }
    for (int i = 0; i < batch_size * num_heads * seq_len * seq_len; i++) {
        h_attention_scores[i] = 0.0f;
    }

    // 分配设备内存
    float *d_Q, *d_K, *d_V, *d_attn_output, *d_attention_scores;
    d_Q = (float *)zerollm_backend::malloc(batch_size * seq_len * embed_dim * sizeof(float));
    d_K = (float *)zerollm_backend::malloc(batch_size * seq_len * embed_dim * sizeof(float));
    d_V = (float *)zerollm_backend::malloc(batch_size * seq_len * embed_dim * sizeof(float));
    d_attn_output = (float *)zerollm_backend::malloc(batch_size * seq_len * embed_dim * sizeof(float));
    d_attention_scores = (float *)zerollm_backend::malloc(batch_size * num_heads * seq_len * seq_len * sizeof(float));

    // 将数据从主机复制到设备
    zerollm_backend::memcpy(d_Q, h_Q, batch_size * seq_len * embed_dim * sizeof(float), zerollm_backend::CopyKind::H2D);
    zerollm_backend::memcpy(d_K, h_K, batch_size * seq_len * embed_dim * sizeof(float), zerollm_backend::CopyKind::H2D);
    zerollm_backend::memcpy(d_V, h_V, batch_size * seq_len * embed_dim * sizeof(float), zerollm_backend::CopyKind::H2D);
    zerollm_backend::memcpy(d_attn_output, h_attn_output, batch_size * seq_len * embed_dim * sizeof(float), zerollm_backend::CopyKind::H2D);
    zerollm_backend::memcpy(d_attention_scores, h_attention_scores, batch_size * num_heads * seq_len * seq_len * sizeof(float), zerollm_backend::CopyKind::H2D);

    // 执行 MHA forward 操作 (因果模式)
    mha_forward<float>(d_Q, d_K, d_V, d_attn_output, d_attention_scores,
                       nullptr, // mask
                       batch_size, seq_len, num_heads, head_dim,
                       true, // is_causal
                       0);

    // 将结果从设备复制回主机
    zerollm_backend::memcpy(h_attn_output, d_attn_output, batch_size * seq_len * embed_dim * sizeof(float), zerollm_backend::CopyKind::D2H);
    zerollm_backend::memcpy(h_attention_scores, d_attention_scores, batch_size * num_heads * seq_len * seq_len * sizeof(float), zerollm_backend::CopyKind::D2H);

    // 验证因果掩码是否正确应用
    // 在因果模式下，注意力分数矩阵的上三角部分（不包括对角线）应该为接近0或负无穷
    bool success = true;
    
    // 检查注意力分数矩阵的第一个head
    float *head_scores = h_attention_scores; // 第一个head的分数
    
    // 对于因果注意力，在seq_len=3的情况下，位置(0,1)、(0,2)和(1,2)应该被掩码
    // 检查这些位置是否接近0（因为kernel中将masked位置设为-exp_inf，然后exp(-inf)=0）
    if (fabsf(head_scores[0 * seq_len + 1]) > 1e-4 ||  // 位置 (0,1)
        fabsf(head_scores[0 * seq_len + 2]) > 1e-4 ||  // 位置 (0,2)
        fabsf(head_scores[1 * seq_len + 2]) > 1e-4) {  // 位置 (1,2)
        success = false;
        std::cerr << "Causal mask not properly applied!" << std::endl;
        std::cerr << "Scores (should be ~0): (" << head_scores[1] << ", " 
                  << head_scores[2] << ", " << head_scores[5] << ")" << std::endl;
    }

    if (success) {
        std::cout << "MHA forward test with causal mask passed!" << std::endl;
    } else {
        std::cerr << "MHA forward test with causal mask failed!" << std::endl;
    }

    // 释放内存
    zerollm_backend::free(d_Q);
    zerollm_backend::free(d_K);
    zerollm_backend::free(d_V);
    zerollm_backend::free(d_attn_output);
    zerollm_backend::free(d_attention_scores);
    free(h_Q);
    free(h_K);
    free(h_V);
    free(h_attn_output);
    free(h_attention_scores);
}

void test_mha_backward() {
    std::cout << "Running MHA backward test..." << std::endl;
    
    const int64_t batch_size = 1;
    const int64_t seq_len = 3;
    const int64_t num_heads = 1;
    const int64_t head_dim = 2;
    const int64_t embed_dim = num_heads * head_dim; // 2
    
    // 输入张量 [batch_size, seq_len, embed_dim]
    float *h_Q = (float*)malloc(batch_size * seq_len * embed_dim * sizeof(float));
    float *h_K = (float*)malloc(batch_size * seq_len * embed_dim * sizeof(float));
    float *h_V = (float*)malloc(batch_size * seq_len * embed_dim * sizeof(float));
    
    // 前向传播的输出
    float *h_attention_scores = (float*)malloc(batch_size * num_heads * seq_len * seq_len * sizeof(float));
    
    // 反向传播的输入梯度
    float *h_d_attn_output = (float*)malloc(batch_size * seq_len * embed_dim * sizeof(float));
    
    // 反向传播的输出梯度
    float *h_d_Q = (float*)malloc(batch_size * seq_len * embed_dim * sizeof(float));
    float *h_d_K = (float*)malloc(batch_size * seq_len * embed_dim * sizeof(float));
    float *h_d_V = (float*)malloc(batch_size * seq_len * embed_dim * sizeof(float));
    
    // 初始化输入数据
    srand(54321); // 固定种子以确保结果可重现
    init_matrix(h_Q, batch_size * seq_len * embed_dim, 1.0f);
    init_matrix(h_K, batch_size * seq_len * embed_dim, 1.0f);
    init_matrix(h_V, batch_size * seq_len * embed_dim, 1.0f);
    
    // 初始化输出梯度
    init_matrix(h_d_attn_output, batch_size * seq_len * embed_dim, 0.1f);
    
    // 清零梯度输出
    for (int i = 0; i < batch_size * seq_len * embed_dim; i++) {
        h_d_Q[i] = 0.0f;
        h_d_K[i] = 0.0f;
        h_d_V[i] = 0.0f;
    }
    
    // 首先执行前向传播以获得注意力分数
    // 分配设备内存
    float *d_Q, *d_K, *d_V, *d_attn_output, *d_attention_scores;
    float *d_d_attn_output, *d_d_Q, *d_d_K, *d_d_V;
    
    d_Q = (float *)zerollm_backend::malloc(batch_size * seq_len * embed_dim * sizeof(float));
    d_K = (float *)zerollm_backend::malloc(batch_size * seq_len * embed_dim * sizeof(float));
    d_V = (float *)zerollm_backend::malloc(batch_size * seq_len * embed_dim * sizeof(float));
    d_attn_output = (float *)zerollm_backend::malloc(batch_size * seq_len * embed_dim * sizeof(float));
    d_attention_scores = (float *)zerollm_backend::malloc(batch_size * num_heads * seq_len * seq_len * sizeof(float));
    d_d_attn_output = (float *)zerollm_backend::malloc(batch_size * seq_len * embed_dim * sizeof(float));
    d_d_Q = (float *)zerollm_backend::malloc(batch_size * seq_len * embed_dim * sizeof(float));
    d_d_K = (float *)zerollm_backend::malloc(batch_size * seq_len * embed_dim * sizeof(float));
    d_d_V = (float *)zerollm_backend::malloc(batch_size * seq_len * embed_dim * sizeof(float));

    // 将数据从主机复制到设备
    zerollm_backend::memcpy(d_Q, h_Q, batch_size * seq_len * embed_dim * sizeof(float), zerollm_backend::CopyKind::H2D);
    zerollm_backend::memcpy(d_K, h_K, batch_size * seq_len * embed_dim * sizeof(float), zerollm_backend::CopyKind::H2D);
    zerollm_backend::memcpy(d_V, h_V, batch_size * seq_len * embed_dim * sizeof(float), zerollm_backend::CopyKind::H2D);
    zerollm_backend::memcpy(d_attention_scores, h_attention_scores, batch_size * num_heads * seq_len * seq_len * sizeof(float), zerollm_backend::CopyKind::H2D);
    zerollm_backend::memcpy(d_d_attn_output, h_d_attn_output, batch_size * seq_len * embed_dim * sizeof(float), zerollm_backend::CopyKind::H2D);
    zerollm_backend::memcpy(d_d_Q, h_d_Q, batch_size * seq_len * embed_dim * sizeof(float), zerollm_backend::CopyKind::H2D);
    zerollm_backend::memcpy(d_d_K, h_d_K, batch_size * seq_len * embed_dim * sizeof(float), zerollm_backend::CopyKind::H2D);
    zerollm_backend::memcpy(d_d_V, h_d_V, batch_size * seq_len * embed_dim * sizeof(float), zerollm_backend::CopyKind::H2D);

    // 执行前向传播
    mha_forward<float>(d_Q, d_K, d_V, d_attn_output, d_attention_scores,
                       nullptr, // mask
                       batch_size, seq_len, num_heads, head_dim,
                       false, // is_causal
                       0);
    
    // 将注意力分数复制回主机以供反向传播使用
    zerollm_backend::memcpy(h_attention_scores, d_attention_scores, batch_size * num_heads * seq_len * seq_len * sizeof(float), zerollm_backend::CopyKind::D2H);
    zerollm_backend::memcpy(d_attention_scores, h_attention_scores, batch_size * num_heads * seq_len * seq_len * sizeof(float), zerollm_backend::CopyKind::H2D);

    // 执行 MHA backward 操作
    mha_backward<float>(d_d_attn_output, d_Q, d_K, d_V, d_attention_scores,
                        nullptr, // mask
                        d_d_Q, d_d_K, d_d_V,
                        batch_size, seq_len, num_heads, head_dim,
                        false, // is_causal
                        0);

    // 将梯度结果从设备复制回主机
    zerollm_backend::memcpy(h_d_Q, d_d_Q, batch_size * seq_len * embed_dim * sizeof(float), zerollm_backend::CopyKind::D2H);
    zerollm_backend::memcpy(h_d_K, d_d_K, batch_size * seq_len * embed_dim * sizeof(float), zerollm_backend::CopyKind::D2H);
    zerollm_backend::memcpy(h_d_V, d_d_V, batch_size * seq_len * embed_dim * sizeof(float), zerollm_backend::CopyKind::D2H);

    // 验证梯度不为零（基本功能检查）
    bool success = true;
    bool found_nonzero_dQ = false;
    bool found_nonzero_dK = false;
    bool found_nonzero_dV = false;
    
    for (int i = 0; i < batch_size * seq_len * embed_dim; i++) {
        if (fabsf(h_d_Q[i]) > 1e-6) {
            found_nonzero_dQ = true;
        }
        if (fabsf(h_d_K[i]) > 1e-6) {
            found_nonzero_dK = true;
        }
        if (fabsf(h_d_V[i]) > 1e-6) {
            found_nonzero_dV = true;
        }
    }

    if (!found_nonzero_dQ) {
        success = false;
        std::cerr << "MHA backward d_Q is all zeros!" << std::endl;
    }
    
    if (!found_nonzero_dK) {
        success = false;
        std::cerr << "MHA backward d_K is all zeros!" << std::endl;
    }
    
    if (!found_nonzero_dV) {
        success = false;
        std::cerr << "MHA backward d_V is all zeros!" << std::endl;
    }

    if (success) {
        std::cout << "MHA backward test passed!" << std::endl;
    } else {
        std::cerr << "MHA backward test failed!" << std::endl;
    }

    // 释放内存
    zerollm_backend::free(d_Q);
    zerollm_backend::free(d_K);
    zerollm_backend::free(d_V);
    zerollm_backend::free(d_attn_output);
    zerollm_backend::free(d_attention_scores);
    zerollm_backend::free(d_d_attn_output);
    zerollm_backend::free(d_d_Q);
    zerollm_backend::free(d_d_K);
    zerollm_backend::free(d_d_V);
    free(h_Q);
    free(h_K);
    free(h_V);
    free(h_attention_scores);
    free(h_d_attn_output);
    free(h_d_Q);
    free(h_d_K);
    free(h_d_V);
}

int main() {
    std::cout << "Starting CUDA MHA Kernel Tests..." << std::endl;
    srand(time(nullptr));
    
    try {
        test_mha_forward();
        std::cout << std::endl;
        
        test_mha_forward_causal();
        std::cout << std::endl;
        
        test_mha_backward();
        std::cout << std::endl;
        
        std::cout << "All MHA tests completed!" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Test failed with exception: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}