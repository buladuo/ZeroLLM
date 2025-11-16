#include <iostream>
#include <cmath>
#include <cstdlib>
#include <ctime>

// 包含 CUDA 运行时
#include <cuda_runtime.h>

// 包含我们要测试的 matmul kernel 函数
#include "kernel/cuda/matmul.cuh"
#include "config.hpp"

#define EPSILON 1e-3f

bool compare_float(float a, float b) {
    return fabsf(a - b) < EPSILON;
}

void init_matrix(float* matrix, int rows, int cols, float factor = 1.0f) {
    for (int i = 0; i < rows * cols; i++) {
        matrix[i] = static_cast<float>(rand()) / RAND_MAX * factor;
    }
}

void init_identity_matrix(float* matrix, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            matrix[i * n + j] = (i == j) ? 1.0f : 0.0f;
        }
    }
}

void init_sequential_matrix(float* matrix, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            matrix[i * cols + j] = static_cast<float>(i * cols + j);
        }
    }
}

void cpu_matmul(const float* A, const float* B, float* C, int M, int N, int K) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

void test_basic_matmul() {
    std::cout << "Running basic matmul test..." << std::endl;
    
    const int M = 3, N = 4, K = 2;
    float *h_A = (float*)malloc(M * K * sizeof(float));
    float *h_B = (float*)malloc(K * N * sizeof(float));
    float *h_C = (float*)malloc(M * N * sizeof(float));
    float *expected = (float*)malloc(M * N * sizeof(float));
    
    // 初始化数据
    // A = [[1, 2],
    //      [3, 4],
    //      [5, 6]]
    for (int i = 0; i < M * K; i++) {
        h_A[i] = static_cast<float>(i + 1);
    }
    
    // B = [[1, 2, 3, 4],
    //      [5, 6, 7, 8]]
    for (int i = 0; i < K * N; i++) {
        h_B[i] = static_cast<float>(i + 1);
    }
    
    // Expected result:
    // C = [[11, 14, 17, 20],
    //      [23, 30, 37, 44],
    //      [35, 46, 57, 68]]
    float expected_vals[] = {11, 14, 17, 20, 23, 30, 37, 44, 35, 46, 57, 68};
    for (int i = 0; i < M * N; i++) {
        expected[i] = expected_vals[i];
    }

    // 分配设备内存
    float *d_A, *d_B, *d_C;
    d_A = (float *)zerollm_backend::malloc(M * K * sizeof(float));
    d_B = (float *)zerollm_backend::malloc(K * N * sizeof(float));
    d_C = (float *)zerollm_backend::malloc(M * N * sizeof(float));

    // 将数据从主机复制到设备
    zerollm_backend::memcpy(d_A, h_A, M * K * sizeof(float), zerollm_backend::CopyKind::H2D);
    zerollm_backend::memcpy(d_B, h_B, K * N * sizeof(float), zerollm_backend::CopyKind::H2D);

    // 执行 matmul 操作
    matmul(d_A, d_B, d_C, M, N, K, 0);

    // 将结果从设备复制回主机
    zerollm_backend::memcpy(h_C, d_C, M * N * sizeof(float), zerollm_backend::CopyKind::D2H);

    // 验证结果
    bool success = true;
    for (int i = 0; i < M * N; ++i) {
        if (!compare_float(h_C[i], expected[i])) {
            success = false;
            std::cerr << "Mismatch at index " << i << ": expected " << expected[i] 
                      << ", got " << h_C[i] << std::endl;
        }
    }

    if (success) {
        std::cout << "Basic matmul test passed!" << std::endl;
    } else {
        std::cerr << "Basic matmul test failed!" << std::endl;
    }

    // 释放内存
    zerollm_backend::free(d_A);
    zerollm_backend::free(d_B);
    zerollm_backend::free(d_C);
    free(h_A);
    free(h_B);
    free(h_C);
    free(expected);
}

void test_identity_matmul() {
    std::cout << "Running identity matrix matmul test..." << std::endl;
    
    const int N = 4;
    float *h_A = (float*)malloc(N * N * sizeof(float));
    float *h_I = (float*)malloc(N * N * sizeof(float));  // Identity matrix
    float *h_C = (float*)malloc(N * N * sizeof(float));

    // 初始化数据 - A为随机矩阵，I为单位矩阵
    init_matrix(h_A, N, N, 10.0f);
    init_identity_matrix(h_I, N);

    // 分配设备内存
    float *d_A, *d_I, *d_C;
    d_A = (float *)zerollm_backend::malloc(N * N * sizeof(float));
    d_I = (float *)zerollm_backend::malloc(N * N * sizeof(float));
    d_C = (float *)zerollm_backend::malloc(N * N * sizeof(float));

    // 将数据从主机复制到设备
    zerollm_backend::memcpy(d_A, h_A, N * N * sizeof(float), zerollm_backend::CopyKind::H2D);
    zerollm_backend::memcpy(d_I, h_I, N * N * sizeof(float), zerollm_backend::CopyKind::H2D);

    // 执行 matmul 操作 (A * I = A)
    matmul(d_A, d_I, d_C, N, N, N, 0);

    // 将结果从设备复制回主机
    zerollm_backend::memcpy(h_C, d_C, N * N * sizeof(float), zerollm_backend::CopyKind::D2H);

    // 验证结果 - 应该与原始矩阵A相同
    bool success = true;
    for (int i = 0; i < N * N; ++i) {
        if (!compare_float(h_C[i], h_A[i])) {
            success = false;
            std::cerr << "Mismatch at index " << i << ": expected " << h_A[i] 
                      << ", got " << h_C[i] << std::endl;
        }
    }

    if (success) {
        std::cout << "Identity matrix matmul test passed!" << std::endl;
    } else {
        std::cerr << "Identity matrix matmul test failed!" << std::endl;
    }

    // 释放内存
    zerollm_backend::free(d_A);
    zerollm_backend::free(d_I);
    zerollm_backend::free(d_C);
    free(h_A);
    free(h_I);
    free(h_C);
}

void test_square_matmul() {
    std::cout << "Running square matrices matmul test..." << std::endl;
    
    const int N = 8;
    float *h_A = (float*)malloc(N * N * sizeof(float));
    float *h_B = (float*)malloc(N * N * sizeof(float));
    float *h_C = (float*)malloc(N * N * sizeof(float));
    float *expected = (float*)malloc(N * N * sizeof(float));

    // 初始化数据
    init_sequential_matrix(h_A, N, N);
    init_sequential_matrix(h_B, N, N);
    
    // 计算期望结果
    cpu_matmul(h_A, h_B, expected, N, N, N);

    // 分配设备内存
    float *d_A, *d_B, *d_C;
    d_A = (float *)zerollm_backend::malloc(N * N * sizeof(float));
    d_B = (float *)zerollm_backend::malloc(N * N * sizeof(float));
    d_C = (float *)zerollm_backend::malloc(N * N * sizeof(float));

    // 将数据从主机复制到设备
    zerollm_backend::memcpy(d_A, h_A, N * N * sizeof(float), zerollm_backend::CopyKind::H2D);
    zerollm_backend::memcpy(d_B, h_B, N * N * sizeof(float), zerollm_backend::CopyKind::H2D);

    // 执行 matmul 操作
    matmul(d_A, d_B, d_C, N, N, N, 0);

    // 将结果从设备复制回主机
    zerollm_backend::memcpy(h_C, d_C, N * N * sizeof(float), zerollm_backend::CopyKind::D2H);

    // 验证结果
    bool success = true;
    for (int i = 0; i < N * N; ++i) {
        if (!compare_float(h_C[i], expected[i])) {
            success = false;
            // 只打印前几个错误以避免输出过多
            if (i < 10) {
                std::cerr << "Mismatch at index " << i << ": expected " << expected[i] 
                          << ", got " << h_C[i] << std::endl;
            }
        }
    }

    if (success) {
        std::cout << "Square matrices matmul test passed!" << std::endl;
    } else {
        std::cerr << "Square matrices matmul test failed!" << std::endl;
    }

    // 释放内存
    zerollm_backend::free(d_A);
    zerollm_backend::free(d_B);
    zerollm_backend::free(d_C);
    free(h_A);
    free(h_B);
    free(h_C);
    free(expected);
}

void test_tiled_matmul() {
    std::cout << "Running tiled matmul test..." << std::endl;
    
    const int M = 32, N = 32, K = 32;
    float *h_A = (float*)malloc(M * K * sizeof(float));
    float *h_B = (float*)malloc(K * N * sizeof(float));
    float *h_C = (float*)malloc(M * N * sizeof(float));
    float *expected = (float*)malloc(M * N * sizeof(float));

    // 初始化数据
    srand(0); // 固定种子以便结果可重现
    init_matrix(h_A, M, K, 5.0f);
    init_matrix(h_B, K, N, 5.0f);
    
    // 计算期望结果
    cpu_matmul(h_A, h_B, expected, M, N, K);

    // 分配设备内存
    float *d_A, *d_B, *d_C;
    d_A = (float *)zerollm_backend::malloc(M * K * sizeof(float));
    d_B = (float *)zerollm_backend::malloc(K * N * sizeof(float));
    d_C = (float *)zerollm_backend::malloc(M * N * sizeof(float));

    // 将数据从主机复制到设备
    zerollm_backend::memcpy(d_A, h_A, M * K * sizeof(float), zerollm_backend::CopyKind::H2D);
    zerollm_backend::memcpy(d_B, h_B, K * N * sizeof(float), zerollm_backend::CopyKind::H2D);

    // 执行 tiled matmul 操作
    matmul_tiled(d_A, d_B, d_C, M, N, K, 0);

    // 将结果从设备复制回主机
    zerollm_backend::memcpy(h_C, d_C, M * N * sizeof(float), zerollm_backend::CopyKind::D2H);

    // 验证结果
    bool success = true;
    int error_count = 0;
    for (int i = 0; i < M * N; ++i) {
        if (!compare_float(h_C[i], expected[i])) {
            success = false;
            error_count++;
            // 只打印前几个错误以避免输出过多
            if (error_count < 5) {
                std::cerr << "Mismatch at (" << i/N << "," << i%N << "): expected " << expected[i] 
                          << ", got " << h_C[i] << std::endl;
            }
        }
    }

    if (success) {
        std::cout << "Tiled matmul test passed!" << std::endl;
    } else {
        std::cerr << "Tiled matmul test failed with " << error_count << " errors!" << std::endl;
    }

    // 释放内存
    zerollm_backend::free(d_A);
    zerollm_backend::free(d_B);
    zerollm_backend::free(d_C);
    free(h_A);
    free(h_B);
    free(h_C);
    free(expected);
}


void cpu_matmul_transposed_A_T_B(const float* A, const float* B, float* C, int M, int N, int K) {
    // Calculate C = A^T * B where A is MxK, A^T is KxM, B is MxN, C is KxN
    for (int i = 0; i < K; i++) {         // Row index for A^T and C
        for (int j = 0; j < N; j++) {     // Column index for B and C
            float sum = 0.0f;
            for (int k = 0; k < M; k++) { // Summation index
                sum += A[k * K + i] * B[k * N + j];  // A^T[i][k] = A[k][i]
            }
            C[i * N + j] = sum;
        }
    }
}

void cpu_matmul_transposed_A_B_T(const float* A, const float* B, float* C, int M, int N, int K) {
    // Calculate C = A * B^T where A is MxK, B is NxK, B^T is KxN, C is MxN
    for (int i = 0; i < M; i++) {         // Row index for A and C
        for (int j = 0; j < N; j++) {     // Column index for B^T and C  
            float sum = 0.0f;
            for (int k = 0; k < K; k++) { // Summation index
                sum += A[i * K + k] * B[j * K + k];  // A[i][k] * B^T[k][j]
            }
            C[i * N + j] = sum;
        }
    }
}


void test_transposed_A_T_B() {
    std::cout << "Running transposed A^T * B matmul test..." << std::endl;
    
    const int M = 4, N = 3, K = 5;  // A is MxK, B is KxN, C is KxN
    float *h_A = (float*)malloc(M * K * sizeof(float));
    float *h_B = (float*)malloc(K * N * sizeof(float));
    float *h_C = (float*)malloc(K * N * sizeof(float));
    float *expected = (float*)malloc(K * N * sizeof(float));

    // Initialize data with sequential values
    init_sequential_matrix(h_A, M, K);
    init_sequential_matrix(h_B, K, N);
    
    // Calculate expected result: C = A^T * B
    cpu_matmul_transposed_A_T_B(h_A, h_B, expected, M, N, K);

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    d_A = (float *)zerollm_backend::malloc(M * K * sizeof(float));
    d_B = (float *)zerollm_backend::malloc(K * N * sizeof(float));
    d_C = (float *)zerollm_backend::malloc(K * N * sizeof(float));

    // Copy data from host to device
    zerollm_backend::memcpy(d_A, h_A, M * K * sizeof(float), zerollm_backend::CopyKind::H2D);
    zerollm_backend::memcpy(d_B, h_B, K * N * sizeof(float), zerollm_backend::CopyKind::H2D);

    // Execute matmul operation: C = A^T * B
    matmul_transposed_A_T_B(d_A, d_B, d_C, M, N, K, 0);

    // Copy result back from device to host
    zerollm_backend::memcpy(h_C, d_C, K * N * sizeof(float), zerollm_backend::CopyKind::D2H);

    // Verify results
    bool success = true;
    int error_count = 0;
    for (int i = 0; i < K * N; ++i) {
        if (!compare_float(h_C[i], expected[i])) {
            success = false;
            error_count++;
            // Print only first few errors to avoid excessive output
            if (error_count < 5) {
                std::cerr << "Mismatch at (" << i/N << "," << i%N << "): expected " << expected[i] 
                          << ", got " << h_C[i] << std::endl;
            }
        }
    }

    if (success) {
        std::cout << "Transposed A^T * B matmul test passed!" << std::endl;
    } else {
        std::cerr << "Transposed A^T * B matmul test failed with " << error_count << " errors!" << std::endl;
    }

    // Free memory
    zerollm_backend::free(d_A);
    zerollm_backend::free(d_B);
    zerollm_backend::free(d_C);
    free(h_A);
    free(h_B);
    free(h_C);
    free(expected);
}


void test_transposed_A_B_T() {
    std::cout << "Running transposed A * B^T matmul test..." << std::endl;
    
    const int M = 4, N = 3, K = 5;  // A is MxK, B is NxK, C is MxN
    float *h_A = (float*)malloc(M * K * sizeof(float));
    float *h_B = (float*)malloc(N * K * sizeof(float));  // Note: B is NxK in memory
    float *h_C = (float*)malloc(M * N * sizeof(float));
    float *expected = (float*)malloc(M * N * sizeof(float));

    // Initialize data with sequential values
    init_sequential_matrix(h_A, M, K);
    init_sequential_matrix(h_B, N, K);  // B is NxK in memory
    
    // Calculate expected result: C = A * B^T
    cpu_matmul_transposed_A_B_T(h_A, h_B, expected, M, N, K);

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    d_A = (float *)zerollm_backend::malloc(M * K * sizeof(float));
    d_B = (float *)zerollm_backend::malloc(N * K * sizeof(float));  // B is NxK in memory
    d_C = (float *)zerollm_backend::malloc(M * N * sizeof(float));

    // Copy data from host to device
    zerollm_backend::memcpy(d_A, h_A, M * K * sizeof(float), zerollm_backend::CopyKind::H2D);
    zerollm_backend::memcpy(d_B, h_B, N * K * sizeof(float), zerollm_backend::CopyKind::H2D);

    // Execute matmul operation: C = A * B^T
    matmul_transposed_A_B_T(d_A, d_B, d_C, M, N, K, 0);

    // Copy result back from device to host
    zerollm_backend::memcpy(h_C, d_C, M * N * sizeof(float), zerollm_backend::CopyKind::D2H);

    // Verify results
    bool success = true;
    int error_count = 0;
    for (int i = 0; i < M * N; ++i) {
        if (!compare_float(h_C[i], expected[i])) {
            success = false;
            error_count++;
            // Print only first few errors to avoid excessive output
            if (error_count < 5) {
                std::cerr << "Mismatch at (" << i/N << "," << i%N << "): expected " << expected[i] 
                          << ", got " << h_C[i] << std::endl;
            }
        }
    }

    if (success) {
        std::cout << "Transposed A * B^T matmul test passed!" << std::endl;
    } else {
        std::cerr << "Transposed A * B^T matmul test failed with " << error_count << " errors!" << std::endl;
    }

    // Free memory
    zerollm_backend::free(d_A);
    zerollm_backend::free(d_B);
    zerollm_backend::free(d_C);
    free(h_A);
    free(h_B);
    free(h_C);
    free(expected);
}

void test_transposed_tiled_A_B_T() {
    std::cout << "Running tiled transposed A * B^T matmul test..." << std::endl;
    
    const int M = 32, N = 32, K = 32;  // A is MxK, B is NxK, C is MxN
    float *h_A = (float*)malloc(M * K * sizeof(float));
    float *h_B = (float*)malloc(N * K * sizeof(float));  // Note: B is NxK in memory
    float *h_C = (float*)malloc(M * N * sizeof(float));
    float *expected = (float*)malloc(M * N * sizeof(float));

    // Initialize data
    srand(1); // Fixed seed for reproducible results
    init_matrix(h_A, M, K, 5.0f);
    init_matrix(h_B, N, K, 5.0f);  // B is NxK in memory
    
    // Calculate expected result: C = A * B^T
    cpu_matmul_transposed_A_B_T(h_A, h_B, expected, M, N, K);

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    d_A = (float *)zerollm_backend::malloc(M * K * sizeof(float));
    d_B = (float *)zerollm_backend::malloc(N * K * sizeof(float));  // B is NxK in memory
    d_C = (float *)zerollm_backend::malloc(M * N * sizeof(float));

    // Copy data from host to device
    zerollm_backend::memcpy(d_A, h_A, M * K * sizeof(float), zerollm_backend::CopyKind::H2D);
    zerollm_backend::memcpy(d_B, h_B, N * K * sizeof(float), zerollm_backend::CopyKind::H2D);

    // Execute tiled matmul operation: C = A * B^T
    matmul_transposed_tiled_A_B_T(d_A, d_B, d_C, M, N, K, 0);

    // Copy result back from device to host
    zerollm_backend::memcpy(h_C, d_C, M * N * sizeof(float), zerollm_backend::CopyKind::D2H);

    // Verify results
    bool success = true;
    int error_count = 0;
    for (int i = 0; i < M * N; ++i) {
        if (!compare_float(h_C[i], expected[i])) {
            success = false;
            error_count++;
            // Print only first few errors to avoid excessive output
            if (error_count < 5) {
                std::cerr << "Mismatch at (" << i/N << "," << i%N << "): expected " << expected[i] 
                          << ", got " << h_C[i] << std::endl;
            }
        }
    }

    if (success) {
        std::cout << "Tiled transposed A * B^T matmul test passed!" << std::endl;
    } else {
        std::cerr << "Tiled transposed A * B^T matmul test failed with " << error_count << " errors!" << std::endl;
    }

    // Free memory
    zerollm_backend::free(d_A);
    zerollm_backend::free(d_B);
    zerollm_backend::free(d_C);
    free(h_A);
    free(h_B);
    free(h_C);
    free(expected);
}

void test_transposed_tiled_A_T_B() {
    std::cout << "Running tiled transposed A^T * B matmul test..." << std::endl;
    
    const int M = 32, N = 32, K = 32;  // A is MxK, B is KxN, C is KxN
    float *h_A = (float*)malloc(M * K * sizeof(float));
    float *h_B = (float*)malloc(K * N * sizeof(float));
    float *h_C = (float*)malloc(K * N * sizeof(float));
    float *expected = (float*)malloc(K * N * sizeof(float));

    // Initialize data
    srand(2); // Fixed seed for reproducible results
    init_matrix(h_A, M, K, 5.0f);
    init_matrix(h_B, K, N, 5.0f);
    
    // Calculate expected result: C = A^T * B
    cpu_matmul_transposed_A_T_B(h_A, h_B, expected, M, N, K);

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    d_A = (float *)zerollm_backend::malloc(M * K * sizeof(float));
    d_B = (float *)zerollm_backend::malloc(K * N * sizeof(float));
    d_C = (float *)zerollm_backend::malloc(K * N * sizeof(float));

    // Copy data from host to device
    zerollm_backend::memcpy(d_A, h_A, M * K * sizeof(float), zerollm_backend::CopyKind::H2D);
    zerollm_backend::memcpy(d_B, h_B, K * N * sizeof(float), zerollm_backend::CopyKind::H2D);

    // Execute tiled matmul operation: C = A^T * B
    matmul_transposed_tiled_A_T_B(d_A, d_B, d_C, M, N, K, 0);

    // Copy result back from device to host
    zerollm_backend::memcpy(h_C, d_C, K * N * sizeof(float), zerollm_backend::CopyKind::D2H);

    // Verify results
    bool success = true;
    int error_count = 0;
    for (int i = 0; i < K * N; ++i) {
        if (!compare_float(h_C[i], expected[i])) {
            success = false;
            error_count++;
            // Print only first few errors to avoid excessive output
            if (error_count < 5) {
                std::cerr << "Mismatch at (" << i/N << "," << i%N << "): expected " << expected[i] 
                          << ", got " << h_C[i] << std::endl;
            }
        }
    }

    if (success) {
        std::cout << "Tiled transposed A^T * B matmul test passed!" << std::endl;
    } else {
        std::cerr << "Tiled transposed A^T * B matmul test failed with " << error_count << " errors!" << std::endl;
    }

    // Free memory
    zerollm_backend::free(d_A);
    zerollm_backend::free(d_B);
    zerollm_backend::free(d_C);
    free(h_A);
    free(h_B);
    free(h_C);
    free(expected);
}

int main() {
    std::cout << "Starting CUDA MatMul Kernel Tests..." << std::endl;
    srand(time(nullptr));
    
    try {
        test_basic_matmul();
        std::cout << std::endl;
        
        test_identity_matmul();
        std::cout << std::endl;
        
        test_square_matmul();
        std::cout << std::endl;
        
        test_tiled_matmul();
        std::cout << std::endl;
        
        test_transposed_A_T_B();
        std::cout << std::endl;
        
        test_transposed_A_B_T();
        std::cout << std::endl;
        
        test_transposed_tiled_A_B_T();
        std::cout << std::endl;
        
        test_transposed_tiled_A_T_B();
        std::cout << std::endl;
        
        std::cout << "All matmul tests completed!" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Test failed with exception: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}