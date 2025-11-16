#include <iostream>
#include <cmath>
#include <cstdlib>
#include <ctime>

// 包含 CUDA 运行时
#include <cuda_runtime.h>

// 包含我们要测试的 add kernel 函数
#include "kernel/cuda/add.cuh"
#include "config.hpp"

#define EPSILON 1e-6f

bool compare_float(float a, float b) {
    return fabsf(a - b) < EPSILON;
}

void test_basic_add() {
    std::cout << "Running basic add test..." << std::endl;
    
    const int size = 10;
    float *h_a = (float*)malloc(size * sizeof(float));
    float *h_b = (float*)malloc(size * sizeof(float));
    float *expected = (float*)malloc(size * sizeof(float));
    float *h_output = (float*)malloc(size * sizeof(float));
    
    // 初始化数据
    float a_vals[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f};
    float b_vals[] = {10.0f, 9.0f, 8.0f, 7.0f, 6.0f, 5.0f, 4.0f, 3.0f, 2.0f, 1.0f};
    float exp_vals[] = {11.0f, 11.0f, 11.0f, 11.0f, 11.0f, 11.0f, 11.0f, 11.0f, 11.0f, 11.0f};
    
    for (int i = 0; i < size; i++) {
        h_a[i] = a_vals[i];
        h_b[i] = b_vals[i];
        expected[i] = exp_vals[i];
    }

    // 分配设备内存
    float *d_a, *d_b, *d_output;
    d_a = (float *)zerollm_backend::malloc(size * sizeof(float));
    d_b = (float *)zerollm_backend::malloc(size * sizeof(float));
    d_output = (float *)zerollm_backend::malloc(size * sizeof(float));

    // 将数据从主机复制到设备
    zerollm_backend::memcpy(d_a, h_a, size * sizeof(float), zerollm_backend::CopyKind::H2D);
    zerollm_backend::memcpy(d_b, h_b, size * sizeof(float), zerollm_backend::CopyKind::H2D);

    // 执行 add 操作
    add(d_a, d_b, d_output, size, 0);

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
        std::cout << "Basic add test passed!" << std::endl;
    } else {
        std::cerr << "Basic add test failed!" << std::endl;
    }

    // 释放内存
    zerollm_backend::free(d_a);
    zerollm_backend::free(d_b);
    zerollm_backend::free(d_output);
    free(h_a);
    free(h_b);
    free(expected);
    free(h_output);
}

void test_large_vector_add() {
    std::cout << "Running large vector add test..." << std::endl;
    
    const int size = 1000000; // 1M elements
    float *h_a = (float*)malloc(size * sizeof(float));
    float *h_b = (float*)malloc(size * sizeof(float));
    float *expected = (float*)malloc(size * sizeof(float));
    float *h_output = (float*)malloc(size * sizeof(float));

    // 初始化输入数据
    for (int i = 0; i < size; ++i) {
        h_a[i] = static_cast<float>(i % 1000);
        h_b[i] = static_cast<float>((i + 1) % 1000);
        expected[i] = h_a[i] + h_b[i];
    }

    // 分配设备内存
    float *d_a, *d_b, *d_output;
    d_a = (float *)zerollm_backend::malloc(size * sizeof(float));
    d_b = (float *)zerollm_backend::malloc(size * sizeof(float));
    d_output = (float *)zerollm_backend::malloc(size * sizeof(float));

    // 将数据从主机复制到设备
    zerollm_backend::memcpy(d_a, h_a, size * sizeof(float), zerollm_backend::CopyKind::H2D);
    zerollm_backend::memcpy(d_b, h_b, size * sizeof(float), zerollm_backend::CopyKind::H2D);

    // 执行 add 操作
    add(d_a, d_b, d_output, size, 0);

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
    srand(time(nullptr));
    for (int i = 0; i < 100; ++i) {
        int idx = rand() % size;
        if (!compare_float(h_output[idx], expected[idx])) {
            success = false;
            std::cerr << "Mismatch at index " << idx << ": expected " << expected[idx] 
                      << ", got " << h_output[idx] << std::endl;
        }
    }

    if (success) {
        std::cout << "Large vector add test passed!" << std::endl;
    } else {
        std::cerr << "Large vector add test failed!" << std::endl;
    }

    // 释放内存
    zerollm_backend::free(d_a);
    zerollm_backend::free(d_b);
    zerollm_backend::free(d_output);
    free(h_a);
    free(h_b);
    free(expected);
    free(h_output);
}

void test_edge_cases() {
    std::cout << "Running edge cases test..." << std::endl;
    
    // 测试大小为1的向量
    {
        const int size = 1;
        float *h_a = (float*)malloc(size * sizeof(float));
        float *h_b = (float*)malloc(size * sizeof(float));
        float *expected = (float*)malloc(size * sizeof(float));
        float *h_output = (float*)malloc(size * sizeof(float));
        
        h_a[0] = 3.14f;
        h_b[0] = 2.71f;
        expected[0] = 5.85f;

        float *d_a, *d_b, *d_output;
        d_a = (float *)zerollm_backend::malloc(size * sizeof(float));
        d_b = (float *)zerollm_backend::malloc(size * sizeof(float));
        d_output = (float *)zerollm_backend::malloc(size * sizeof(float));

        zerollm_backend::memcpy(d_a, h_a, size * sizeof(float), zerollm_backend::CopyKind::H2D);
        zerollm_backend::memcpy(d_b, h_b, size * sizeof(float), zerollm_backend::CopyKind::H2D);

        add(d_a, d_b, d_output, size, 0);

        zerollm_backend::memcpy(h_output, d_output, size * sizeof(float), zerollm_backend::CopyKind::D2H);

        bool success = compare_float(h_output[0], expected[0]);
        if (!success) {
            std::cerr << "Edge case (size=1) failed: expected " << expected[0] 
                      << ", got " << h_output[0] << std::endl;
        }

        zerollm_backend::free(d_a);
        zerollm_backend::free(d_b);
        zerollm_backend::free(d_output);
        free(h_a);
        free(h_b);
        free(expected);
        free(h_output);

        if (success) {
            std::cout << "Edge case (size=1) passed!" << std::endl;
        } else {
            std::cerr << "Edge case (size=1) failed!" << std::endl;
        }
    }

    // 测试负数
    {
        const int size = 5;
        float *h_a = (float*)malloc(size * sizeof(float));
        float *h_b = (float*)malloc(size * sizeof(float));
        float *expected = (float*)malloc(size * sizeof(float));
        float *h_output = (float*)malloc(size * sizeof(float));
        
        float a_vals[] = {-1.0f, -2.0f, -3.0f, -4.0f, -5.0f};
        float b_vals[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
        float exp_vals[] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        
        for (int i = 0; i < size; i++) {
            h_a[i] = a_vals[i];
            h_b[i] = b_vals[i];
            expected[i] = exp_vals[i];
        }

        float *d_a, *d_b, *d_output;
        d_a = (float *)zerollm_backend::malloc(size * sizeof(float));
        d_b = (float *)zerollm_backend::malloc(size * sizeof(float));
        d_output = (float *)zerollm_backend::malloc(size * sizeof(float));

        zerollm_backend::memcpy(d_a, h_a, size * sizeof(float), zerollm_backend::CopyKind::H2D);
        zerollm_backend::memcpy(d_b, h_b, size * sizeof(float), zerollm_backend::CopyKind::H2D);

        add(d_a, d_b, d_output, size, 0);

        zerollm_backend::memcpy(h_output, d_output, size * sizeof(float), zerollm_backend::CopyKind::D2H);

        bool success = true;
        for (int i = 0; i < size; ++i) {
            if (!compare_float(h_output[i], expected[i])) {
                success = false;
                std::cerr << "Edge case (negative numbers) failed at index " << i 
                          << ": expected " << expected[i] << ", got " << h_output[i] << std::endl;
            }
        }

        zerollm_backend::free(d_a);
        zerollm_backend::free(d_b);
        zerollm_backend::free(d_output);
        free(h_a);
        free(h_b);
        free(expected);
        free(h_output);

        if (success) {
            std::cout << "Edge case (negative numbers) passed!" << std::endl;
        } else {
            std::cerr << "Edge case (negative numbers) failed!" << std::endl;
        }
    }
}

void test_add_inplace() {
    std::cout << "Running add_inplace test..." << std::endl;
    
    const int size = 6;
    float *h_a = (float*)malloc(size * sizeof(float));
    float *h_b = (float*)malloc(size * sizeof(float));
    float *expected = (float*)malloc(size * sizeof(float));
    
    float a_vals[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    float b_vals[] = {6.0f, 5.0f, 4.0f, 3.0f, 2.0f, 1.0f};
    float exp_vals[] = {7.0f, 7.0f, 7.0f, 7.0f, 7.0f, 7.0f};
    
    for (int i = 0; i < size; i++) {
        h_a[i] = a_vals[i];
        h_b[i] = b_vals[i];
        expected[i] = exp_vals[i];
    }

    float *d_a, *d_b;
    d_a = (float *)zerollm_backend::malloc(size * sizeof(float));
    d_b = (float *)zerollm_backend::malloc(size * sizeof(float));

    zerollm_backend::memcpy(d_a, h_a, size * sizeof(float), zerollm_backend::CopyKind::H2D);
    zerollm_backend::memcpy(d_b, h_b, size * sizeof(float), zerollm_backend::CopyKind::H2D);

    add_inplace(d_a, d_b, size, 0);

    float *h_output = (float*)malloc(size * sizeof(float));
    zerollm_backend::memcpy(h_output, d_a, size * sizeof(float), zerollm_backend::CopyKind::D2H);

    bool success = true;
    for (int i = 0; i < size; ++i) {
        if (!compare_float(h_output[i], expected[i])) {
            success = false;
            std::cerr << "Inplace add failed at index " << i << ": expected " << expected[i] 
                      << ", got " << h_output[i] << std::endl;
        }
    }

    if (success) {
        std::cout << "Add_inplace test passed!" << std::endl;
    } else {
        std::cerr << "Add_inplace test failed!" << std::endl;
    }

    zerollm_backend::free(d_a);
    zerollm_backend::free(d_b);
    free(h_a);
    free(h_b);
    free(expected);
    free(h_output);
}

int main() {
    std::cout << "Starting CUDA Add Kernel Tests..." << std::endl;
    
    try {
        test_basic_add();
        std::cout << std::endl;
        
        test_large_vector_add();
        std::cout << std::endl;
        
        test_edge_cases();
        std::cout << std::endl;
        
        test_add_inplace();
        std::cout << std::endl;
        
        std::cout << "All tests completed!" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Test failed with exception: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}