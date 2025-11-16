#include <iostream>
#include <cmath>
#include <cstdlib>
#include <ctime>

// 包含 CUDA 运行时
#include <cuda_runtime.h>

// 包含我们要测试的 transpose kernel 函数
#include "kernel/cuda/transpose.cuh"
#include "config.hpp"

#define EPSILON 1e-6f

bool compare_float(float a, float b) {
    return fabsf(a - b) < EPSILON;
}

void test_basic_transpose() {
    std::cout << "Running basic transpose test..." << std::endl;
    
    const int rows = 3, cols = 4;
    float *h_input = (float*)malloc(rows * cols * sizeof(float));
    float *h_output = (float*)malloc(cols * rows * sizeof(float));
    float *expected = (float*)malloc(cols * rows * sizeof(float));
    
    // 初始化输入数据
    // 输入矩阵:
    // [1, 2, 3, 4]
    // [5, 6, 7, 8]
    // [9, 10, 11, 12]
    for (int i = 0; i < rows * cols; i++) {
        h_input[i] = static_cast<float>(i + 1);
    }
    
    // 期望输出矩阵:
    // [1, 5, 9]
    // [2, 6, 10]
    // [3, 7, 11]
    // [4, 8, 12]
    float expected_vals[] = {1, 5, 9, 2, 6, 10, 3, 7, 11, 4, 8, 12};
    for (int i = 0; i < cols * rows; i++) {
        expected[i] = expected_vals[i];
    }

    // 分配设备内存
    float *d_input, *d_output;
    d_input = (float *)zerollm_backend::malloc(rows * cols * sizeof(float));
    d_output = (float *)zerollm_backend::malloc(cols * rows * sizeof(float));

    // 将数据从主机复制到设备
    zerollm_backend::memcpy(d_input, h_input, rows * cols * sizeof(float), zerollm_backend::CopyKind::H2D);

    // 执行 transpose 操作
    transpose(d_input, d_output, rows, cols, 0);

    // 将结果从设备复制回主机
    zerollm_backend::memcpy(h_output, d_output, cols * rows * sizeof(float), zerollm_backend::CopyKind::D2H);

    // 验证结果
    bool success = true;
    for (int i = 0; i < cols * rows; ++i) {
        if (!compare_float(h_output[i], expected[i])) {
            success = false;
            std::cerr << "Mismatch at index " << i << ": expected " << expected[i] 
                      << ", got " << h_output[i] << std::endl;
        }
    }

    if (success) {
        std::cout << "Basic transpose test passed!" << std::endl;
    } else {
        std::cerr << "Basic transpose test failed!" << std::endl;
    }

    // 释放内存
    zerollm_backend::free(d_input);
    zerollm_backend::free(d_output);
    free(h_input);
    free(h_output);
    free(expected);
}

void test_square_transpose() {
    std::cout << "Running square matrix transpose test..." << std::endl;
    
    const int rows = 4, cols = 4;
    float *h_input = (float*)malloc(rows * cols * sizeof(float));
    float *h_output = (float*)malloc(cols * rows * sizeof(float));
    float *expected = (float*)malloc(cols * rows * sizeof(float));
    
    // 初始化输入数据 - 顺序填充
    for (int i = 0; i < rows * cols; i++) {
        h_input[i] = static_cast<float>(i);
    }
    
    // 期望输出矩阵 (手动计算转置)
    // 输入:
    // [0, 1, 2, 3]
    // [4, 5, 6, 7]
    // [8, 9, 10, 11]
    // [12, 13, 14, 15]
    // 
    // 输出:
    // [0, 4, 8, 12]
    // [1, 5, 9, 13]
    // [2, 6, 10, 14]
    // [3, 7, 11, 15]
    float expected_vals[] = {0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15};
    for (int i = 0; i < cols * rows; i++) {
        expected[i] = expected_vals[i];
    }

    // 分配设备内存
    float *d_input, *d_output;
    d_input = (float *)zerollm_backend::malloc(rows * cols * sizeof(float));
    d_output = (float *)zerollm_backend::malloc(cols * rows * sizeof(float));

    // 将数据从主机复制到设备
    zerollm_backend::memcpy(d_input, h_input, rows * cols * sizeof(float), zerollm_backend::CopyKind::H2D);

    // 执行 transpose 操作
    transpose(d_input, d_output, rows, cols, 0);

    // 将结果从设备复制回主机
    zerollm_backend::memcpy(h_output, d_output, cols * rows * sizeof(float), zerollm_backend::CopyKind::D2H);

    // 验证结果
    bool success = true;
    for (int i = 0; i < cols * rows; ++i) {
        if (!compare_float(h_output[i], expected[i])) {
            success = false;
            std::cerr << "Mismatch at index " << i << ": expected " << expected[i] 
                      << ", got " << h_output[i] << std::endl;
        }
    }

    if (success) {
        std::cout << "Square matrix transpose test passed!" << std::endl;
    } else {
        std::cerr << "Square matrix transpose test failed!" << std::endl;
    }

    // 释放内存
    zerollm_backend::free(d_input);
    zerollm_backend::free(d_output);
    free(h_input);
    free(h_output);
    free(expected);
}

void test_vector_transpose() {
    std::cout << "Running vector transpose test..." << std::endl;
    
    // 测试行向量转置为列向量
    const int rows = 1, cols = 5;
    float *h_input = (float*)malloc(rows * cols * sizeof(float));
    float *h_output = (float*)malloc(cols * rows * sizeof(float));
    float *expected = (float*)malloc(cols * rows * sizeof(float));
    
    // 初始化输入数据
    for (int i = 0; i < rows * cols; i++) {
        h_input[i] = static_cast<float>(i * 2);
    }
    
    // 对于向量，转置应该保持值不变，只是形状改变
    for (int i = 0; i < cols * rows; i++) {
        expected[i] = h_input[i];
    }

    // 分配设备内存
    float *d_input, *d_output;
    d_input = (float *)zerollm_backend::malloc(rows * cols * sizeof(float));
    d_output = (float *)zerollm_backend::malloc(cols * rows * sizeof(float));

    // 将数据从主机复制到设备
    zerollm_backend::memcpy(d_input, h_input, rows * cols * sizeof(float), zerollm_backend::CopyKind::H2D);

    // 执行 transpose 操作
    transpose(d_input, d_output, rows, cols, 0);

    // 将结果从设备复制回主机
    zerollm_backend::memcpy(h_output, d_output, cols * rows * sizeof(float), zerollm_backend::CopyKind::D2H);

    // 验证结果
    bool success = true;
    for (int i = 0; i < cols * rows; ++i) {
        if (!compare_float(h_output[i], expected[i])) {
            success = false;
            std::cerr << "Mismatch at index " << i << ": expected " << expected[i] 
                      << ", got " << h_output[i] << std::endl;
        }
    }

    if (success) {
        std::cout << "Vector transpose test passed!" << std::endl;
    } else {
        std::cerr << "Vector transpose test failed!" << std::endl;
    }

    // 释放内存
    zerollm_backend::free(d_input);
    zerollm_backend::free(d_output);
    free(h_input);
    free(h_output);
    free(expected);
}

void test_large_matrix_transpose() {
    std::cout << "Running large matrix transpose test..." << std::endl;
    
    const int rows = 128, cols = 256;
    float *h_input = (float*)malloc(rows * cols * sizeof(float));
    float *h_output = (float*)malloc(cols * rows * sizeof(float));
    float *expected = (float*)malloc(cols * rows * sizeof(float));

    // 初始化输入数据
    for (int i = 0; i < rows * cols; i++) {
        h_input[i] = static_cast<float>(i % 1000) / 100.0f;
    }
    
    // 计算期望输出
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            expected[j * rows + i] = h_input[i * cols + j];
        }
    }

    // 分配设备内存
    float *d_input, *d_output;
    d_input = (float *)zerollm_backend::malloc(rows * cols * sizeof(float));
    d_output = (float *)zerollm_backend::malloc(cols * rows * sizeof(float));

    // 将数据从主机复制到设备
    zerollm_backend::memcpy(d_input, h_input, rows * cols * sizeof(float), zerollm_backend::CopyKind::H2D);

    // 执行 transpose 操作
    transpose(d_input, d_output, rows, cols, 0);

    // 将结果从设备复制回主机
    zerollm_backend::memcpy(h_output, d_output, cols * rows * sizeof(float), zerollm_backend::CopyKind::D2H);

    // 验证结果 (只检查部分元素)
    bool success = true;
    for (int i = 0; i < 100; ++i) { // 只检查前100个元素
        if (!compare_float(h_output[i], expected[i])) {
            success = false;
            std::cerr << "Mismatch at index " << i << ": expected " << expected[i] 
                      << ", got " << h_output[i] << std::endl;
        }
    }

    // 再随机检查一些元素
    srand(time(nullptr));
    for (int i = 0; i < 100; ++i) {
        int idx = rand() % (cols * rows);
        if (!compare_float(h_output[idx], expected[idx])) {
            success = false;
            std::cerr << "Mismatch at index " << idx << ": expected " << expected[idx] 
                      << ", got " << h_output[idx] << std::endl;
        }
    }

    if (success) {
        std::cout << "Large matrix transpose test passed!" << std::endl;
    } else {
        std::cerr << "Large matrix transpose test failed!" << std::endl;
    }

    // 释放内存
    zerollm_backend::free(d_input);
    zerollm_backend::free(d_output);
    free(h_input);
    free(h_output);
    free(expected);
}

int main() {
    std::cout << "Starting CUDA Transpose Kernel Tests..." << std::endl;
    srand(time(nullptr));
    
    try {
        test_basic_transpose();
        std::cout << std::endl;
        
        test_square_transpose();
        std::cout << std::endl;
        
        test_vector_transpose();
        std::cout << std::endl;
        
        test_large_matrix_transpose();
        std::cout << std::endl;
        
        std::cout << "All transpose tests completed!" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Test failed with exception: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}