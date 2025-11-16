#include <iostream>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <vector>

// 包含 CUDA 运行时
#include <cuda_runtime.h>

// 包含我们要测试的 softmax kernel 函数
#include "kernel/cuda/softmax.cuh"
#include "config.hpp"

#define EPSILON 1e-5f

bool compare_float(float a, float b) {
    return fabsf(a - b) < EPSILON;
}

void cpu_softmax(const float* input, float* output, int M, int N) {
    for (int i = 0; i < M; i++) {
        // 找到该行的最大值（数值稳定性）
        float max_val = input[i * N];
        for (int j = 1; j < N; j++) {
            if (input[i * N + j] > max_val) {
                max_val = input[i * N + j];
            }
        }
        
        // 计算exp(x - max)并求和
        float sum = 0.0f;
        for (int j = 0; j < N; j++) {
            output[i * N + j] = expf(input[i * N + j] - max_val);
            sum += output[i * N + j];
        }
        
        // 归一化
        for (int j = 0; j < N; j++) {
            output[i * N + j] /= sum;
        }
    }
}

void test_basic_softmax() {
    std::cout << "Running basic softmax test..." << std::endl;
    
    const int M = 2, N = 3;
    float *h_input = (float*)malloc(M * N * sizeof(float));
    float *h_output = (float*)malloc(M * N * sizeof(float));
    float *expected = (float*)malloc(M * N * sizeof(float));
    
    // 初始化输入数据
    // 第一行: [1.0, 2.0, 3.0]
    // 第二行: [0.0, -1.0, 1.0]
    float input_vals[] = {1.0f, 2.0f, 3.0f, 0.0f, -1.0f, 1.0f};
    for (int i = 0; i < M * N; i++) {
        h_input[i] = input_vals[i];
    }
    
    // 计算期望输出
    cpu_softmax(h_input, expected, M, N);

    // 分配设备内存
    float *d_input, *d_output;
    d_input = (float *)zerollm_backend::malloc(M * N * sizeof(float));
    d_output = (float *)zerollm_backend::malloc(M * N * sizeof(float));

    // 将数据从主机复制到设备
    zerollm_backend::memcpy(d_input, h_input, M * N * sizeof(float), zerollm_backend::CopyKind::H2D);

    // 执行 softmax 操作
    softmax(d_input, d_output, M, N, 0);

    // 将结果从设备复制回主机
    zerollm_backend::memcpy(h_output, d_output, M * N * sizeof(float), zerollm_backend::CopyKind::D2H);

    // 验证结果
    bool success = true;
    for (int i = 0; i < M; i++) {
        float row_sum = 0.0f;
        for (int j = 0; j < N; j++) {
            row_sum += h_output[i * N + j];
            if (!compare_float(h_output[i * N + j], expected[i * N + j])) {
                success = false;
                std::cerr << "Mismatch at (" << i << ", " << j << "): expected " << expected[i * N + j] 
                          << ", got " << h_output[i * N + j] << std::endl;
            }
        }
        // 检查每行和是否为1
        if (!compare_float(row_sum, 1.0f)) {
            success = false;
            std::cerr << "Row sum for row " << i << " is not 1.0: " << row_sum << std::endl;
        }
    }

    if (success) {
        std::cout << "Basic softmax test passed!" << std::endl;
    } else {
        std::cerr << "Basic softmax test failed!" << std::endl;
    }

    // 释放内存
    zerollm_backend::free(d_input);
    zerollm_backend::free(d_output);
    free(h_input);
    free(h_output);
    free(expected);
}

void test_single_row_softmax() {
    std::cout << "Running single row softmax test..." << std::endl;
    
    const int M = 1, N = 5;
    float *h_input = (float*)malloc(M * N * sizeof(float));
    float *h_output = (float*)malloc(M * N * sizeof(float));
    float *expected = (float*)malloc(M * N * sizeof(float));
    
    // 初始化输入数据
    float input_vals[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    for (int i = 0; i < M * N; i++) {
        h_input[i] = input_vals[i];
    }
    
    // 计算期望输出
    cpu_softmax(h_input, expected, M, N);

    // 分配设备内存
    float *d_input, *d_output;
    d_input = (float *)zerollm_backend::malloc(M * N * sizeof(float));
    d_output = (float *)zerollm_backend::malloc(M * N * sizeof(float));

    // 将数据从主机复制到设备
    zerollm_backend::memcpy(d_input, h_input, M * N * sizeof(float), zerollm_backend::CopyKind::H2D);

    // 执行 softmax 操作
    softmax(d_input, d_output, M, N, 0);

    // 将结果从设备复制回主机
    zerollm_backend::memcpy(h_output, d_output, M * N * sizeof(float), zerollm_backend::CopyKind::D2H);

    // 验证结果
    bool success = true;
    float row_sum = 0.0f;
    for (int j = 0; j < N; j++) {
        row_sum += h_output[j];
        if (!compare_float(h_output[j], expected[j])) {
            success = false;
            std::cerr << "Mismatch at index " << j << ": expected " << expected[j] 
                      << ", got " << h_output[j] << std::endl;
        }
    }
    // 检查行和是否为1
    if (!compare_float(row_sum, 1.0f)) {
        success = false;
        std::cerr << "Row sum is not 1.0: " << row_sum << std::endl;
    }

    if (success) {
        std::cout << "Single row softmax test passed!" << std::endl;
    } else {
        std::cerr << "Single row softmax test failed!" << std::endl;
    }

    // 释放内存
    zerollm_backend::free(d_input);
    zerollm_backend::free(d_output);
    free(h_input);
    free(h_output);
    free(expected);
}

void test_large_softmax() {
    std::cout << "Running large softmax test..." << std::endl;
    
    const int M = 10, N = 1000;
    float *h_input = (float*)malloc(M * N * sizeof(float));
    float *h_output = (float*)malloc(M * N * sizeof(float));
    float *expected = (float*)malloc(M * N * sizeof(float));

    // 初始化输入数据
    srand(12345); // 固定种子以确保结果可重现
    for (int i = 0; i < M * N; i++) {
        h_input[i] = (static_cast<float>(rand()) / RAND_MAX) * 10.0f - 5.0f; // [-5, 5]
    }
    
    // 计算期望输出
    cpu_softmax(h_input, expected, M, N);

    // 分配设备内存
    float *d_input, *d_output;
    d_input = (float *)zerollm_backend::malloc(M * N * sizeof(float));
    d_output = (float *)zerollm_backend::malloc(M * N * sizeof(float));

    // 将数据从主机复制到设备
    zerollm_backend::memcpy(d_input, h_input, M * N * sizeof(float), zerollm_backend::CopyKind::H2D);

    // 执行 softmax 操作
    softmax(d_input, d_output, M, N, 0);

    // 将结果从设备复制回主机
    zerollm_backend::memcpy(h_output, d_output, M * N * sizeof(float), zerollm_backend::CopyKind::D2H);

    // 验证结果 (只检查部分元素)
    bool success = true;
    for (int i = 0; i < M; i++) {
        float row_sum = 0.0f;
        for (int j = 0; j < N; j++) {
            row_sum += h_output[i * N + j];
        }
        // 检查每行和是否为1
        if (!compare_float(row_sum, 1.0f)) {
            success = false;
            std::cerr << "Row sum for row " << i << " is not 1.0: " << row_sum << std::endl;
        }
        
        // 检查几个特定元素
        for (int j = 0; j < 10 && j < N; j++) {
            if (!compare_float(h_output[i * N + j], expected[i * N + j])) {
                success = false;
                std::cerr << "Mismatch at (" << i << ", " << j << "): expected " << expected[i * N + j] 
                          << ", got " << h_output[i * N + j] << std::endl;
            }
        }
    }

    if (success) {
        std::cout << "Large softmax test passed!" << std::endl;
    } else {
        std::cerr << "Large softmax test failed!" << std::endl;
    }

    // 释放内存
    zerollm_backend::free(d_input);
    zerollm_backend::free(d_output);
    free(h_input);
    free(h_output);
    free(expected);
}

void test_edge_cases() {
    std::cout << "Running edge cases softmax test..." << std::endl;
    
    const int M = 3, N = 4;
    float *h_input = (float*)malloc(M * N * sizeof(float));
    float *h_output = (float*)malloc(M * N * sizeof(float));
    float *expected = (float*)malloc(M * N * sizeof(float));
    
    // 测试边缘情况：
    // 1. 全零行
    // 2. 很大的数值
    // 3. 很小的数值
    float input_vals[] = {
        0.0f, 0.0f, 0.0f, 0.0f,     // 全零行
        100.0f, 100.0f, 100.0f, 100.0f, // 大数值行
        -100.0f, -100.0f, -100.0f, -100.0f // 小数值行
    };
    for (int i = 0; i < M * N; i++) {
        h_input[i] = input_vals[i];
    }
    
    // 计算期望输出
    cpu_softmax(h_input, expected, M, N);

    // 分配设备内存
    float *d_input, *d_output;
    d_input = (float *)zerollm_backend::malloc(M * N * sizeof(float));
    d_output = (float *)zerollm_backend::malloc(M * N * sizeof(float));

    // 将数据从主机复制到设备
    zerollm_backend::memcpy(d_input, h_input, M * N * sizeof(float), zerollm_backend::CopyKind::H2D);

    // 执行 softmax 操作
    softmax(d_input, d_output, M, N, 0);

    // 将结果从设备复制回主机
    zerollm_backend::memcpy(h_output, d_output, M * N * sizeof(float), zerollm_backend::CopyKind::D2H);

    // 验证结果
    bool success = true;
    for (int i = 0; i < M; i++) {
        float row_sum = 0.0f;
        for (int j = 0; j < N; j++) {
            row_sum += h_output[i * N + j];
            if (!compare_float(h_output[i * N + j], expected[i * N + j])) {
                success = false;
                std::cerr << "Mismatch at (" << i << ", " << j << "): expected " << expected[i * N + j] 
                          << ", got " << h_output[i * N + j] << std::endl;
            }
        }
        // 检查每行和是否为1
        if (!compare_float(row_sum, 1.0f)) {
            success = false;
            std::cerr << "Row sum for row " << i << " is not 1.0: " << row_sum << std::endl;
        }
    }

    if (success) {
        std::cout << "Edge cases softmax test passed!" << std::endl;
    } else {
        std::cerr << "Edge cases softmax test failed!" << std::endl;
    }

    // 释放内存
    zerollm_backend::free(d_input);
    zerollm_backend::free(d_output);
    free(h_input);
    free(h_output);
    free(expected);
}

int main() {
    std::cout << "Starting CUDA Softmax Kernel Tests..." << std::endl;
    srand(time(nullptr));
    
    try {
        test_basic_softmax();
        std::cout << std::endl;
        
        test_single_row_softmax();
        std::cout << std::endl;
        
        test_large_softmax();
        std::cout << std::endl;
        
        test_edge_cases();
        std::cout << std::endl;
        
        std::cout << "All softmax tests completed!" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Test failed with exception: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}