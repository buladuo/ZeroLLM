#include <iostream>
#include <cmath>
#include <cstdlib>
#include <ctime>

// 包含 CUDA 运行时
#include <cuda_runtime.h>

// 包含我们要测试的 relu kernel 函数
#include "kernel/cuda/relu.cuh"
#include "config.hpp"

#define EPSILON 1e-6f

bool compare_float(float a, float b) {
    return fabsf(a - b) < EPSILON;
}

void test_relu_forward_positive_values() {
    std::cout << "Running ReLU forward test with positive values..." << std::endl;
    
    const int size = 10;
    float *h_input = (float*)malloc(size * sizeof(float));
    float *h_output = (float*)malloc(size * sizeof(float));
    float *expected = (float*)malloc(size * sizeof(float));
    
    // 初始化数据 - 全部为正数
    for (int i = 0; i < size; i++) {
        h_input[i] = static_cast<float>(i + 1);  // 1, 2, 3, ..., 10
        expected[i] = static_cast<float>(i + 1); // Same as input since all positive
    }

    // 分配设备内存
    float *d_input, *d_output;
    d_input = (float *)zerollm_backend::malloc(size * sizeof(float));
    d_output = (float *)zerollm_backend::malloc(size * sizeof(float));

    // 将数据从主机复制到设备
    zerollm_backend::memcpy(d_input, h_input, size * sizeof(float), zerollm_backend::CopyKind::H2D);

    // 执行 relu_forward 操作
    relu_forward(d_input, d_output, size, 0);

    // 将结果从设备复制回主机
    zerollm_backend::memcpy(h_output, d_output, size * sizeof(float), zerollm_backend::CopyKind::D2H);

    // 验证结果
    bool success = true;
    for (int i = 0; i < size; ++i) {
        if (!compare_float(h_output[i], expected[i])) {
            success = false;
            std::cerr << "Mismatch at index " << i << ": expected " << expected[i] 
                      << ", got " << h_output[i] << std::endl;
        }
    }

    if (success) {
        std::cout << "ReLU forward test with positive values passed!" << std::endl;
    } else {
        std::cerr << "ReLU forward test with positive values failed!" << std::endl;
    }

    // 释放内存
    zerollm_backend::free(d_input);
    zerollm_backend::free(d_output);
    free(h_input);
    free(h_output);
    free(expected);
}

void test_relu_forward_mixed_values() {
    std::cout << "Running ReLU forward test with mixed values..." << std::endl;
    
    const int size = 10;
    float *h_input = (float*)malloc(size * sizeof(float));
    float *h_output = (float*)malloc(size * sizeof(float));
    float *expected = (float*)malloc(size * sizeof(float));
    
    // 初始化数据 - 正数、负数和零混合
    float input_vals[] = {-3.0f, -1.0f, 0.0f, 1.0f, 2.0f, -0.5f, 4.0f, -2.0f, 0.0f, 5.0f};
    float expected_vals[] = {0.0f, 0.0f, 0.0f, 1.0f, 2.0f, 0.0f, 4.0f, 0.0f, 0.0f, 5.0f};
    
    for (int i = 0; i < size; i++) {
        h_input[i] = input_vals[i];
        expected[i] = expected_vals[i];
    }

    // 分配设备内存
    float *d_input, *d_output;
    d_input = (float *)zerollm_backend::malloc(size * sizeof(float));
    d_output = (float *)zerollm_backend::malloc(size * sizeof(float));

    // 将数据从主机复制到设备
    zerollm_backend::memcpy(d_input, h_input, size * sizeof(float), zerollm_backend::CopyKind::H2D);

    // 执行 relu_forward 操作
    relu_forward(d_input, d_output, size, 0);

    // 将结果从设备复制回主机
    zerollm_backend::memcpy(h_output, d_output, size * sizeof(float), zerollm_backend::CopyKind::D2H);

    // 验证结果
    bool success = true;
    for (int i = 0; i < size; ++i) {
        if (!compare_float(h_output[i], expected[i])) {
            success = false;
            std::cerr << "Mismatch at index " << i << ": expected " << expected[i] 
                      << ", got " << h_output[i] << std::endl;
        }
    }

    if (success) {
        std::cout << "ReLU forward test with mixed values passed!" << std::endl;
    } else {
        std::cerr << "ReLU forward test with mixed values failed!" << std::endl;
    }

    // 释放内存
    zerollm_backend::free(d_input);
    zerollm_backend::free(d_output);
    free(h_input);
    free(h_output);
    free(expected);
}

void test_relu_forward_large_array() {
    std::cout << "Running ReLU forward test with large array..." << std::endl;
    
    const int size = 1000000; // 1M elements
    float *h_input = (float*)malloc(size * sizeof(float));
    float *h_output = (float*)malloc(size * sizeof(float));
    float *expected = (float*)malloc(size * sizeof(float));

    // 初始化数据 - 随机值范围[-10, 10]
    srand(12345); // 固定种子以便结果可重现
    for (int i = 0; i < size; i++) {
        h_input[i] = (static_cast<float>(rand()) / RAND_MAX) * 20.0f - 10.0f;
        expected[i] = h_input[i] > 0 ? h_input[i] : 0.0f;
    }

    // 分配设备内存
    float *d_input, *d_output;
    d_input = (float *)zerollm_backend::malloc(size * sizeof(float));
    d_output = (float *)zerollm_backend::malloc(size * sizeof(float));

    // 将数据从主机复制到设备
    zerollm_backend::memcpy(d_input, h_input, size * sizeof(float), zerollm_backend::CopyKind::H2D);

    // 执行 relu_forward 操作
    relu_forward(d_input, d_output, size, 0);

    // 将结果从设备复制回主机
    zerollm_backend::memcpy(h_output, d_output, size * sizeof(float), zerollm_backend::CopyKind::D2H);

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
    for (int i = 0; i < 100; ++i) {
        int idx = rand() % size;
        if (!compare_float(h_output[idx], expected[idx])) {
            success = false;
            std::cerr << "Mismatch at index " << idx << ": expected " << expected[idx] 
                      << ", got " << h_output[idx] << std::endl;
        }
    }

    if (success) {
        std::cout << "ReLU forward test with large array passed!" << std::endl;
    } else {
        std::cerr << "ReLU forward test with large array failed!" << std::endl;
    }

    // 释放内存
    zerollm_backend::free(d_input);
    zerollm_backend::free(d_output);
    free(h_input);
    free(h_output);
    free(expected);
}

void test_relu_backward() {
    std::cout << "Running ReLU backward test..." << std::endl;
    
    const int size = 6;
    float *h_input = (float*)malloc(size * sizeof(float));
    float *h_grad_output = (float*)malloc(size * sizeof(float));
    float *h_grad_input = (float*)malloc(size * sizeof(float));
    float *expected = (float*)malloc(size * sizeof(float));
    
    // 初始化数据
    float input_vals[] = {-1.0f, 2.0f, -3.0f, 0.0f, 4.0f, -5.0f};
    float grad_output_vals[] = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
    // 根据CUDA内核实现，当输入为0时，梯度应该为0
    float expected_vals[] = {0.0f, 1.0f, 0.0f, 0.0f, 1.0f, 0.0f}; // gradient is passed through if input > 0
    
    for (int i = 0; i < size; i++) {
        h_input[i] = input_vals[i];
        h_grad_output[i] = grad_output_vals[i];
        expected[i] = expected_vals[i];
    }

    // 分配设备内存
    float *d_input, *d_grad_output, *d_grad_input;
    d_input = (float *)zerollm_backend::malloc(size * sizeof(float));
    d_grad_output = (float *)zerollm_backend::malloc(size * sizeof(float));
    d_grad_input = (float *)zerollm_backend::malloc(size * sizeof(float));

    // 将数据从主机复制到设备
    zerollm_backend::memcpy(d_input, h_input, size * sizeof(float), zerollm_backend::CopyKind::H2D);
    zerollm_backend::memcpy(d_grad_output, h_grad_output, size * sizeof(float), zerollm_backend::CopyKind::H2D);

    // 执行 relu_backward 操作
    relu_backward(d_input, d_grad_output, d_grad_input, size, 0);

    // 将结果从设备复制回主机
    zerollm_backend::memcpy(h_grad_input, d_grad_input, size * sizeof(float), zerollm_backend::CopyKind::D2H);

    // 验证结果
    bool success = true;
    for (int i = 0; i < size; ++i) {
        if (!compare_float(h_grad_input[i], expected[i])) {
            success = false;
            std::cerr << "Mismatch at index " << i << ": expected " << expected[i] 
                      << ", got " << h_grad_input[i] << std::endl;
        }
    }

    if (success) {
        std::cout << "ReLU backward test passed!" << std::endl;
    } else {
        std::cerr << "ReLU backward test failed!" << std::endl;
    }

    // 释放内存
    zerollm_backend::free(d_input);
    zerollm_backend::free(d_grad_output);
    zerollm_backend::free(d_grad_input);
    free(h_input);
    free(h_grad_output);
    free(h_grad_input);
    free(expected);
}

void test_relu_backward_complex_gradients() {
    std::cout << "Running ReLU backward test with complex gradients..." << std::endl;
    
    const int size = 8;
    float *h_input = (float*)malloc(size * sizeof(float));
    float *h_grad_output = (float*)malloc(size * sizeof(float));
    float *h_grad_input = (float*)malloc(size * sizeof(float));
    float *expected = (float*)malloc(size * sizeof(float));
    
    // 初始化数据
    float input_vals[] = {1.0f, -2.0f, 3.0f, 0.0f, -4.0f, 5.0f, -0.5f, 0.0f};
    float grad_output_vals[] = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f};
    // 根据CUDA内核实现，当输入为0时，梯度应该为0
    float expected_vals[] = {0.1f, 0.0f, 0.3f, 0.0f, 0.0f, 0.6f, 0.0f, 0.0f};
    
    for (int i = 0; i < size; i++) {
        h_input[i] = input_vals[i];
        h_grad_output[i] = grad_output_vals[i];
        expected[i] = expected_vals[i];
    }

    // 分配设备内存
    float *d_input, *d_grad_output, *d_grad_input;
    d_input = (float *)zerollm_backend::malloc(size * sizeof(float));
    d_grad_output = (float *)zerollm_backend::malloc(size * sizeof(float));
    d_grad_input = (float *)zerollm_backend::malloc(size * sizeof(float));

    // 将数据从主机复制到设备
    zerollm_backend::memcpy(d_input, h_input, size * sizeof(float), zerollm_backend::CopyKind::H2D);
    zerollm_backend::memcpy(d_grad_output, h_grad_output, size * sizeof(float), zerollm_backend::CopyKind::H2D);

    // 执行 relu_backward 操作
    relu_backward(d_input, d_grad_output, d_grad_input, size, 0);

    // 将结果从设备复制回主机
    zerollm_backend::memcpy(h_grad_input, d_grad_input, size * sizeof(float), zerollm_backend::CopyKind::D2H);

    // 验证结果
    bool success = true;
    for (int i = 0; i < size; ++i) {
        if (!compare_float(h_grad_input[i], expected[i])) {
            success = false;
            std::cerr << "Mismatch at index " << i << ": expected " << expected[i] 
                      << ", got " << h_grad_input[i] << std::endl;
        }
    }

    if (success) {
        std::cout << "ReLU backward test with complex gradients passed!" << std::endl;
    } else {
        std::cerr << "ReLU backward test with complex gradients failed!" << std::endl;
    }

    // 释放内存
    zerollm_backend::free(d_input);
    zerollm_backend::free(d_grad_output);
    zerollm_backend::free(d_grad_input);
    free(h_input);
    free(h_grad_output);
    free(h_grad_input);
    free(expected);
}
int main() {
    std::cout << "Starting CUDA ReLU Kernel Tests..." << std::endl;
    srand(time(nullptr));
    
    try {
        test_relu_forward_positive_values();
        std::cout << std::endl;
        
        test_relu_forward_mixed_values();
        std::cout << std::endl;
        
        test_relu_forward_large_array();
        std::cout << std::endl;
        
        test_relu_backward();
        std::cout << std::endl;
        
        test_relu_backward_complex_gradients();
        std::cout << std::endl;
        
        std::cout << "All ReLU tests completed!" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Test failed with exception: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}