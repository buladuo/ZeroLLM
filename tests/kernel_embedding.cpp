#include <iostream>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <vector>

// 包含 CUDA 运行时
#include <cuda_runtime.h>

// 包含我们要测试的 embedding kernel 函数
#include "kernel/cuda/embedding.cuh"
#include "config.hpp"

#define EPSILON 1e-6f

bool compare_float(float a, float b) {
    return fabsf(a - b) < EPSILON;
}

void test_embedding_forward() {
    std::cout << "Running embedding forward test..." << std::endl;
    
    const int batch_size = 2;
    const int seq_len = 3;
    const int embed_dim = 4;
    const int vocab_size = 10;
    const int max_seq_len = 5;
    
    // 输入索引 [batch_size, seq_len]
    int *h_input = (int*)malloc(batch_size * seq_len * sizeof(int));
    // 嵌入表 [vocab_size, embed_dim]
    float *h_embedding_table = (float*)malloc(vocab_size * embed_dim * sizeof(float));
    // 位置编码表 [max_seq_len, embed_dim]
    float *h_pos_encoding_table = (float*)malloc(max_seq_len * embed_dim * sizeof(float));
    // 输出 [batch_size, seq_len, embed_dim]
    float *h_output = (float*)malloc(batch_size * seq_len * embed_dim * sizeof(float));
    // 期望输出
    float *expected = (float*)malloc(batch_size * seq_len * embed_dim * sizeof(float));
    
    // 初始化输入数据
    // batch 0: [1, 2, 3]
    // batch 1: [4, 5, 6]
    int input_vals[] = {1, 2, 3, 4, 5, 6};
    for (int i = 0; i < batch_size * seq_len; i++) {
        h_input[i] = input_vals[i];
    }
    
    // 初始化嵌入表，简单地使用 i*j 作为值
    for (int i = 0; i < vocab_size; i++) {
        for (int j = 0; j < embed_dim; j++) {
            h_embedding_table[i * embed_dim + j] = static_cast<float>(i * embed_dim + j);
        }
    }
    
    // 初始化位置编码表
    for (int i = 0; i < max_seq_len; i++) {
        for (int j = 0; j < embed_dim; j++) {
            h_pos_encoding_table[i * embed_dim + j] = static_cast<float>(i + j * 0.1f);
        }
    }
    
    // 手动计算期望输出
    for (int b = 0; b < batch_size; b++) {
        for (int s = 0; s < seq_len; s++) {
            int token_id = h_input[b * seq_len + s];
            for (int d = 0; d < embed_dim; d++) {
                expected[(b * seq_len + s) * embed_dim + d] = 
                    h_embedding_table[token_id * embed_dim + d] + 
                    h_pos_encoding_table[s * embed_dim + d];
            }
        }
    }

    // 分配设备内存
    int *d_input;
    float *d_embedding_table, *d_pos_encoding_table, *d_output;
    d_input = (int *)zerollm_backend::malloc(batch_size * seq_len * sizeof(int));
    d_embedding_table = (float *)zerollm_backend::malloc(vocab_size * embed_dim * sizeof(float));
    d_pos_encoding_table = (float *)zerollm_backend::malloc(max_seq_len * embed_dim * sizeof(float));
    d_output = (float *)zerollm_backend::malloc(batch_size * seq_len * embed_dim * sizeof(float));

    // 将数据从主机复制到设备
    zerollm_backend::memcpy(d_input, h_input, batch_size * seq_len * sizeof(int), zerollm_backend::CopyKind::H2D);
    zerollm_backend::memcpy(d_embedding_table, h_embedding_table, vocab_size * embed_dim * sizeof(float), zerollm_backend::CopyKind::H2D);
    zerollm_backend::memcpy(d_pos_encoding_table, h_pos_encoding_table, max_seq_len * embed_dim * sizeof(float), zerollm_backend::CopyKind::H2D);

    // 执行 embedding forward 操作
    cuda_embedding_forward<float, int>(d_input, d_embedding_table, d_pos_encoding_table, d_output,
                                      batch_size, seq_len, embed_dim, vocab_size, max_seq_len, 0);

    // 将结果从设备复制回主机
    zerollm_backend::memcpy(h_output, d_output, batch_size * seq_len * embed_dim * sizeof(float), zerollm_backend::CopyKind::D2H);

    // 验证结果
    bool success = true;
    for (int i = 0; i < batch_size * seq_len * embed_dim; ++i) {
        if (!compare_float(h_output[i], expected[i])) {
            success = false;
            std::cerr << "Mismatch at index " << i << ": expected " << expected[i] 
                      << ", got " << h_output[i] << std::endl;
            break;
        }
    }

    if (success) {
        std::cout << "Embedding forward test passed!" << std::endl;
    } else {
        std::cerr << "Embedding forward test failed!" << std::endl;
    }

    // 释放内存
    zerollm_backend::free(d_input);
    zerollm_backend::free(d_embedding_table);
    zerollm_backend::free(d_pos_encoding_table);
    zerollm_backend::free(d_output);
    free(h_input);
    free(h_embedding_table);
    free(h_pos_encoding_table);
    free(h_output);
    free(expected);
}

void test_embedding_backward() {
    std::cout << "Running embedding backward test..." << std::endl;
    
    const int batch_size = 2;
    const int seq_len = 3;
    const int embed_dim = 4;
    const int vocab_size = 10;
    
    // 输入索引 [batch_size, seq_len]
    int *h_input = (int*)malloc(batch_size * seq_len * sizeof(int));
    // 输出梯度 [batch_size, seq_len, embed_dim]
    float *h_d_output = (float*)malloc(batch_size * seq_len * embed_dim * sizeof(float));
    // 嵌入表梯度 [vocab_size, embed_dim]
    float *h_d_embedding_table = (float*)malloc(vocab_size * embed_dim * sizeof(float));
    float *expected_d_embedding_table = (float*)malloc(vocab_size * embed_dim * sizeof(float));
    
    // 初始化输入数据
    // batch 0: [1, 2, 1]
    // batch 1: [2, 1, 2]
    int input_vals[] = {1, 2, 1, 2, 1, 2};
    for (int i = 0; i < batch_size * seq_len; i++) {
        h_input[i] = input_vals[i];
    }
    
    // 初始化输出梯度
    for (int i = 0; i < batch_size * seq_len * embed_dim; i++) {
        h_d_output[i] = static_cast<float>(i % 5 + 1); // 1-5循环
    }
    
    // 初始化嵌入表梯度为0
    for (int i = 0; i < vocab_size * embed_dim; i++) {
        h_d_embedding_table[i] = 0.0f;
        expected_d_embedding_table[i] = 0.0f;
    }
    
    // 手动计算期望的嵌入表梯度
    // 对于每个输入索引，将其对应的梯度累加到期望结果中
    for (int b = 0; b < batch_size; b++) {
        for (int s = 0; s < seq_len; s++) {
            int token_id = h_input[b * seq_len + s];
            for (int d = 0; d < embed_dim; d++) {
                expected_d_embedding_table[token_id * embed_dim + d] += 
                    h_d_output[(b * seq_len + s) * embed_dim + d];
            }
        }
    }

    // 分配设备内存
    int *d_input;
    float *d_d_output, *d_d_embedding_table;
    d_input = (int *)zerollm_backend::malloc(batch_size * seq_len * sizeof(int));
    d_d_output = (float *)zerollm_backend::malloc(batch_size * seq_len * embed_dim * sizeof(float));
    d_d_embedding_table = (float *)zerollm_backend::malloc(vocab_size * embed_dim * sizeof(float));

    // 将数据从主机复制到设备
    zerollm_backend::memcpy(d_input, h_input, batch_size * seq_len * sizeof(int), zerollm_backend::CopyKind::H2D);
    zerollm_backend::memcpy(d_d_output, h_d_output, batch_size * seq_len * embed_dim * sizeof(float), zerollm_backend::CopyKind::H2D);
    zerollm_backend::memcpy(d_d_embedding_table, h_d_embedding_table, vocab_size * embed_dim * sizeof(float), zerollm_backend::CopyKind::H2D);

    // 执行 embedding backward 操作
    cuda_embedding_backward<float, int>(d_input, d_d_output, d_d_embedding_table,
                                       batch_size, seq_len, embed_dim, vocab_size, 0);

    // 将结果从设备复制回主机
    zerollm_backend::memcpy(h_d_embedding_table, d_d_embedding_table, vocab_size * embed_dim * sizeof(float), zerollm_backend::CopyKind::D2H);

    // 验证结果
    bool success = true;
    for (int i = 0; i < vocab_size * embed_dim; ++i) {
        if (!compare_float(h_d_embedding_table[i], expected_d_embedding_table[i])) {
            success = false;
            std::cerr << "Mismatch at index " << i << ": expected " << expected_d_embedding_table[i] 
                      << ", got " << h_d_embedding_table[i] << std::endl;
            break;
        }
    }

    if (success) {
        std::cout << "Embedding backward test passed!" << std::endl;
    } else {
        std::cerr << "Embedding backward test failed!" << std::endl;
    }

    // 释放内存
    zerollm_backend::free(d_input);
    zerollm_backend::free(d_d_output);
    zerollm_backend::free(d_d_embedding_table);
    free(h_input);
    free(h_d_output);
    free(h_d_embedding_table);
    free(expected_d_embedding_table);
}

void test_embedding_with_invalid_indices() {
    std::cout << "Running embedding test with invalid indices..." << std::endl;
    
    const int batch_size = 1;
    const int seq_len = 3;
    const int embed_dim = 2;
    const int vocab_size = 5;
    const int max_seq_len = 3;
    
    // 输入包含无效索引
    int *h_input = (int*)malloc(batch_size * seq_len * sizeof(int));
    float *h_embedding_table = (float*)malloc(vocab_size * embed_dim * sizeof(float));
    float *h_pos_encoding_table = (float*)malloc(max_seq_len * embed_dim * sizeof(float));
    float *h_output = (float*)malloc(batch_size * seq_len * embed_dim * sizeof(float));
    
    // 输入包含无效索引: [-1, 2, 10]
    h_input[0] = -1;   // 无效索引 (< 0)
    h_input[1] = 2;    // 有效索引
    h_input[2] = 10;   // 无效索引 (>= vocab_size)
    
    // 初始化嵌入表
    for (int i = 0; i < vocab_size; i++) {
        for (int j = 0; j < embed_dim; j++) {
            h_embedding_table[i * embed_dim + j] = static_cast<float>(i * embed_dim + j);
        }
    }
    
    // 初始化位置编码表
    for (int i = 0; i < max_seq_len; i++) {
        for (int j = 0; j < embed_dim; j++) {
            h_pos_encoding_table[i * embed_dim + j] = static_cast<float>(i + j * 0.1f);
        }
    }

    // 分配设备内存
    int *d_input;
    float *d_embedding_table, *d_pos_encoding_table, *d_output;
    d_input = (int *)zerollm_backend::malloc(batch_size * seq_len * sizeof(int));
    d_embedding_table = (float *)zerollm_backend::malloc(vocab_size * embed_dim * sizeof(float));
    d_pos_encoding_table = (float *)zerollm_backend::malloc(max_seq_len * embed_dim * sizeof(float));
    d_output = (float *)zerollm_backend::malloc(batch_size * seq_len * embed_dim * sizeof(float));

    // 将数据从主机复制到设备
    zerollm_backend::memcpy(d_input, h_input, batch_size * seq_len * sizeof(int), zerollm_backend::CopyKind::H2D);
    zerollm_backend::memcpy(d_embedding_table, h_embedding_table, vocab_size * embed_dim * sizeof(float), zerollm_backend::CopyKind::H2D);
    zerollm_backend::memcpy(d_pos_encoding_table, h_pos_encoding_table, max_seq_len * embed_dim * sizeof(float), zerollm_backend::CopyKind::H2D);

    // 执行 embedding forward 操作
    cuda_embedding_forward<float, int>(d_input, d_embedding_table, d_pos_encoding_table, d_output,
                                      batch_size, seq_len, embed_dim, vocab_size, max_seq_len, 0);

    // 将结果从设备复制回主机
    zerollm_backend::memcpy(h_output, d_output, batch_size * seq_len * embed_dim * sizeof(float), zerollm_backend::CopyKind::D2H);

    // 验证无效索引位置的输出是否为0（因为内核中会跳过这些索引）
    bool success = true;
    // 检查无效索引 -1 对应的输出是否为0
    for (int d = 0; d < embed_dim; d++) {
        if (h_output[0 * embed_dim + d] != 0.0f) {
            success = false;
            std::cerr << "Invalid index -1 at position 0 produced non-zero output: " << h_output[0 * embed_dim + d] << std::endl;
        }
    }
    
    // 检查有效索引 2 对应的输出是否正确
    for (int d = 0; d < embed_dim; d++) {
        float expected = h_embedding_table[2 * embed_dim + d] + h_pos_encoding_table[1 * embed_dim + d];
        if (!compare_float(h_output[1 * embed_dim + d], expected)) {
            success = false;
            std::cerr << "Valid index 2 at position 1 mismatch: expected " << expected 
                      << ", got " << h_output[1 * embed_dim + d] << std::endl;
        }
    }
    
    // 检查无效索引 10 对应的输出是否为0
    for (int d = 0; d < embed_dim; d++) {
        if (h_output[2 * embed_dim + d] != 0.0f) {
            success = false;
            std::cerr << "Invalid index 10 at position 2 produced non-zero output: " << h_output[2 * embed_dim + d] << std::endl;
        }
    }

    if (success) {
        std::cout << "Embedding test with invalid indices passed!" << std::endl;
    } else {
        std::cerr << "Embedding test with invalid indices failed!" << std::endl;
    }

    // 释放内存
    zerollm_backend::free(d_input);
    zerollm_backend::free(d_embedding_table);
    zerollm_backend::free(d_pos_encoding_table);
    zerollm_backend::free(d_output);
    free(h_input);
    free(h_embedding_table);
    free(h_pos_encoding_table);
    free(h_output);
}

int main() {
    std::cout << "Starting CUDA Embedding Kernel Tests..." << std::endl;
    srand(time(nullptr));
    
    try {
        test_embedding_forward();
        std::cout << std::endl;
        
        test_embedding_backward();
        std::cout << std::endl;
        
        test_embedding_with_invalid_indices();
        std::cout << std::endl;
        
        std::cout << "All embedding tests completed!" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Test failed with exception: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}