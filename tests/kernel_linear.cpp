#include <iostream>
#include <cmath>
#include <cstdlib>
#include <ctime>

// 包含 CUDA 运行时
#include <cuda_runtime.h>

// 包含我们要测试的 linear kernel 函数
#include "kernel/cuda/linear.cuh"
#include "config.hpp"

#define EPSILON 1e-3f

bool compare_float(float a, float b) {
    return fabsf(a - b) < EPSILON;
}

void init_matrix(float* matrix, int rows, int cols, float factor = 1.0f) {
    for (int i = 0; i < rows * cols; i++) {
        matrix[i] = (static_cast<float>(rand()) / RAND_MAX) * 2.0f * factor - factor;
    }
}

void cpu_matmul_transposed(const float* A, const float* B, float* C, int M, int N, int K) {
    // C[M,N] = A[M,K] * B[N,K]^T 
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += A[i * K + k] * B[j * K + k];
            }
            C[i * N + j] = sum;
        }
    }
}

void test_linear_forward_without_bias() {
    std::cout << "Running linear forward test without bias..." << std::endl;
    
    const int64_t M = 2, K = 3, N = 4;
    float *h_input = (float*)malloc(M * K * sizeof(float));
    float *h_weight = (float*)malloc(N * K * sizeof(float));
    float *h_output = (float*)malloc(M * N * sizeof(float));
    float *expected = (float*)malloc(M * N * sizeof(float));
    
    // 初始化数据
    // input: [[1, 2, 3], 
    //         [4, 5, 6]]
    float input_vals[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    for (int i = 0; i < M * K; i++) {
        h_input[i] = input_vals[i];
    }
    
    // weight: [[1, 2, 3],
    //          [4, 5, 6], 
    //          [7, 8, 9],
    //          [10, 11, 12]]
    float weight_vals[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f};
    for (int i = 0; i < N * K; i++) {
        h_weight[i] = weight_vals[i];
    }
    
    // 计算期望输出 (output = input * weight^T)
    // input[0] * weight[0]^T = [1,2,3] * [1,2,3] = 1*1 + 2*2 + 3*3 = 14
    // input[0] * weight[1]^T = [1,2,3] * [4,5,6] = 1*4 + 2*5 + 3*6 = 32
    // input[0] * weight[2]^T = [1,2,3] * [7,8,9] = 1*7 + 2*8 + 3*9 = 50
    // input[0] * weight[3]^T = [1,2,3] * [10,11,12] = 1*10 + 2*11 + 3*12 = 68
    // input[1] * weight[0]^T = [4,5,6] * [1,2,3] = 4*1 + 5*2 + 6*3 = 32
    // input[1] * weight[1]^T = [4,5,6] * [4,5,6] = 4*4 + 5*5 + 6*6 = 77
    // input[1] * weight[2]^T = [4,5,6] * [7,8,9] = 4*7 + 5*8 + 6*9 = 122
    // input[1] * weight[3]^T = [4,5,6] * [10,11,12] = 4*10 + 5*11 + 6*12 = 167
    //
    // Expected: [14, 32, 50, 68, 32, 77, 122, 167]
    cpu_matmul_transposed(h_input, h_weight, expected, M, N, K);

    // 分配设备内存
    float *d_input, *d_weight, *d_output;
    d_input = (float *)zerollm_backend::malloc(M * K * sizeof(float));
    d_weight = (float *)zerollm_backend::malloc(N * K * sizeof(float));
    d_output = (float *)zerollm_backend::malloc(M * N * sizeof(float));

    // 将数据从主机复制到设备
    zerollm_backend::memcpy(d_input, h_input, M * K * sizeof(float), zerollm_backend::CopyKind::H2D);
    zerollm_backend::memcpy(d_weight, h_weight, N * K * sizeof(float), zerollm_backend::CopyKind::H2D);

    // 执行 linear forward 操作 (不使用偏置)
    linear_forward<float>(d_input, d_weight, nullptr, d_output, M, K, N, false, 0);

    // 将结果从设备复制回主机
    zerollm_backend::memcpy(h_output, d_output, M * N * sizeof(float), zerollm_backend::CopyKind::D2H);

    // 验证结果
    bool success = true;
    for (int i = 0; i < M * N; ++i) {
        if (!compare_float(h_output[i], expected[i])) {
            success = false;
            std::cerr << "Mismatch at index " << i << ": expected " << expected[i] 
                      << ", got " << h_output[i] << std::endl;
        }
    }

    if (success) {
        std::cout << "Linear forward test without bias passed!" << std::endl;
    } else {
        std::cerr << "Linear forward test without bias failed!" << std::endl;
    }

    // 释放内存
    zerollm_backend::free(d_input);
    zerollm_backend::free(d_weight);
    zerollm_backend::free(d_output);
    free(h_input);
    free(h_weight);
    free(h_output);
    free(expected);
}

void test_linear_forward_with_bias() {
    std::cout << "Running linear forward test with bias..." << std::endl;
    
    const int64_t M = 2, K = 3, N = 4;
    float *h_input = (float*)malloc(M * K * sizeof(float));
    float *h_weight = (float*)malloc(N * K * sizeof(float));
    float *h_bias = (float*)malloc(N * sizeof(float));
    float *h_output = (float*)malloc(M * N * sizeof(float));
    float *expected = (float*)malloc(M * N * sizeof(float));
    
    // 初始化数据
    // input: [[1, 2, 3], 
    //         [4, 5, 6]]
    float input_vals[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    for (int i = 0; i < M * K; i++) {
        h_input[i] = input_vals[i];
    }
    
    // weight: [[1, 2, 3],
    //          [4, 5, 6], 
    //          [7, 8, 9],
    //          [10, 11, 12]]
    float weight_vals[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f};
    for (int i = 0; i < N * K; i++) {
        h_weight[i] = weight_vals[i];
    }
    
    // bias: [1, 2, 3, 4]
    float bias_vals[] = {1.0f, 2.0f, 3.0f, 4.0f};
    for (int i = 0; i < N; i++) {
        h_bias[i] = bias_vals[i];
    }
    
    // 计算期望输出 (output = input * weight^T + bias)
    cpu_matmul_transposed(h_input, h_weight, expected, M, N, K);
    // 添加偏置
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            expected[i * N + j] += h_bias[j];
        }
    }

    // 分配设备内存
    float *d_input, *d_weight, *d_bias, *d_output;
    d_input = (float *)zerollm_backend::malloc(M * K * sizeof(float));
    d_weight = (float *)zerollm_backend::malloc(N * K * sizeof(float));
    d_bias = (float *)zerollm_backend::malloc(N * sizeof(float));
    d_output = (float *)zerollm_backend::malloc(M * N * sizeof(float));

    // 将数据从主机复制到设备
    zerollm_backend::memcpy(d_input, h_input, M * K * sizeof(float), zerollm_backend::CopyKind::H2D);
    zerollm_backend::memcpy(d_weight, h_weight, N * K * sizeof(float), zerollm_backend::CopyKind::H2D);
    zerollm_backend::memcpy(d_bias, h_bias, N * sizeof(float), zerollm_backend::CopyKind::H2D);

    // 执行 linear forward 操作 (使用偏置)
    linear_forward(d_input, d_weight, d_bias, d_output, M, K, N, true, 0);

    // 将结果从设备复制回主机
    zerollm_backend::memcpy(h_output, d_output, M * N * sizeof(float), zerollm_backend::CopyKind::D2H);

    // 验证结果
    bool success = true;
    for (int i = 0; i < M * N; ++i) {
        if (!compare_float(h_output[i], expected[i])) {
            success = false;
            std::cerr << "Mismatch at index " << i << ": expected " << expected[i] 
                      << ", got " << h_output[i] << std::endl;
        }
    }

    if (success) {
        std::cout << "Linear forward test with bias passed!" << std::endl;
    } else {
        std::cerr << "Linear forward test with bias failed!" << std::endl;
    }

    // 释放内存
    zerollm_backend::free(d_input);
    zerollm_backend::free(d_weight);
    zerollm_backend::free(d_bias);
    zerollm_backend::free(d_output);
    free(h_input);
    free(h_weight);
    free(h_bias);
    free(h_output);
    free(expected);
}

void test_linear_forward_large() {
    std::cout << "Running large linear forward test..." << std::endl;
    
    const int64_t M = 16, K = 128, N = 64;
    float *h_input = (float*)malloc(M * K * sizeof(float));
    float *h_weight = (float*)malloc(N * K * sizeof(float));
    float *h_bias = (float*)malloc(N * sizeof(float));
    float *h_output = (float*)malloc(M * N * sizeof(float));
    float *expected = (float*)malloc(M * N * sizeof(float));

    // 初始化数据
    srand(12345); // 固定种子以确保结果可重现
    init_matrix(h_input, M, K, 1.0f);
    init_matrix(h_weight, N, K, 1.0f);
    init_matrix(h_bias, N, 1, 1.0f);
    
    // 计算期望输出
    cpu_matmul_transposed(h_input, h_weight, expected, M, N, K);
    // 添加偏置
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            expected[i * N + j] += h_bias[j];
        }
    }

    // 分配设备内存
    float *d_input, *d_weight, *d_bias, *d_output;
    d_input = (float *)zerollm_backend::malloc(M * K * sizeof(float));
    d_weight = (float *)zerollm_backend::malloc(N * K * sizeof(float));
    d_bias = (float *)zerollm_backend::malloc(N * sizeof(float));
    d_output = (float *)zerollm_backend::malloc(M * N * sizeof(float));

    // 将数据从主机复制到设备
    zerollm_backend::memcpy(d_input, h_input, M * K * sizeof(float), zerollm_backend::CopyKind::H2D);
    zerollm_backend::memcpy(d_weight, h_weight, N * K * sizeof(float), zerollm_backend::CopyKind::H2D);
    zerollm_backend::memcpy(d_bias, h_bias, N * sizeof(float), zerollm_backend::CopyKind::H2D);

    // 执行 linear forward 操作
    linear_forward(d_input, d_weight, d_bias, d_output, M, K, N, true, 0);

    // 将结果从设备复制回主机
    zerollm_backend::memcpy(h_output, d_output, M * N * sizeof(float), zerollm_backend::CopyKind::D2H);

    // 验证结果 (只检查部分元素)
    bool success = true;
    int error_count = 0;
    for (int i = 0; i < 10; i++) { // 只检查前10个元素
        for (int j = 0; j < 10 && j < N; j++) {
            if (!compare_float(h_output[i * N + j], expected[i * N + j])) {
                success = false;
                error_count++;
                // 只打印前几个错误以避免输出过多
                if (error_count < 5) {
                    std::cerr << "Mismatch at (" << i << ", " << j << "): expected " << expected[i * N + j] 
                              << ", got " << h_output[i * N + j] << std::endl;
                }
            }
        }
    }

    if (success) {
        std::cout << "Large linear forward test passed!" << std::endl;
    } else {
        std::cerr << "Large linear forward test failed with " << error_count << " errors!" << std::endl;
    }

    // 释放内存
    zerollm_backend::free(d_input);
    zerollm_backend::free(d_weight);
    zerollm_backend::free(d_bias);
    zerollm_backend::free(d_output);
    free(h_input);
    free(h_weight);
    free(h_bias);
    free(h_output);
    free(expected);
}

int main() {
    std::cout << "Starting CUDA Linear Kernel Tests..." << std::endl;
    srand(time(nullptr));
    
    try {
        test_linear_forward_without_bias();
        std::cout << std::endl;
        
        test_linear_forward_with_bias();
        std::cout << std::endl;
        
        test_linear_forward_large();
        std::cout << std::endl;
        
        std::cout << "All linear tests completed!" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Test failed with exception: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}