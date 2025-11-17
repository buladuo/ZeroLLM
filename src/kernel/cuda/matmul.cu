#include "matmul.cuh"
#include <iostream>
#include "config.hpp"

// 矩阵乘法: C = A * B
template<typename T>
__global__ void matmul_kernel(const T* A, const T* B, T* C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        T sum = 0;
        for (int k = 0; k < K; ++k) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

// 矩阵转置乘法: C = A^T * B
template<typename T>
__global__ void matmul_transposed_A_T_B_kernel(const T* A, const T* B, T* C, int M, int N, int K) { 
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < K && col < N) {
        T sum = 0;
        for (int k = 0; k < M; ++k) {
            sum += A[k * K + row] * B[k * N + col];  // A的转置
        }
        C[row * N + col] = sum;
    }
}

// 矩阵转置乘法: C = A * B^T
template<typename T>
__global__ void matmul_transposed_A_B_T_kernel(const T* A, const T* B, T* C, int M, int N, int K) { 
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        T sum = 0;
        for (int k = 0; k < K; ++k) {
            sum += A[row * K + k] * B[col * K + k];  // B的转置
        }
        C[row * N + col] = sum;
    }
}

template<typename T>
__global__ void matmul_tiled_kernel(const T* A, const T* B, T* C, int M, int N, int K) {
    const int TILE_SIZE = ZEROLLM_DEFAULT_TILE_SIZE;
    
    // 使用静态声明的共享内存
    __shared__ T As[TILE_SIZE][TILE_SIZE + 1];  // +1 避免 bank 冲突
    __shared__ T Bs[TILE_SIZE][TILE_SIZE + 1];

    // 线程在块内的坐标
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    // C 矩阵的全局坐标
    int global_row = blockIdx.y * TILE_SIZE + ty;
    int global_col = blockIdx.x * TILE_SIZE + tx;

    T Cvalue = 0;

    // 遍历 K 维度的瓦片
    for (int t = 0; t < K; t += TILE_SIZE) {
        // 载入 A 的子矩阵 (A[global_row, t:t+TILE_SIZE])
        if (global_row < M && (t + tx) < K)
            As[ty][tx] = A[global_row * K + t + tx];
        else
            As[ty][tx] = 0;

        // 载入 B 的子矩阵 (B[t:t+TILE_SIZE, global_col])
        if ((t + ty) < K && global_col < N)
            Bs[ty][tx] = B[(t + ty) * N + global_col];
        else
            Bs[ty][tx] = 0;

        __syncthreads();

        // 在共享内存上计算子矩阵乘积
        for (int k = 0; k < TILE_SIZE; ++k) {
            Cvalue += As[ty][k] * Bs[k][tx];
        }

        __syncthreads();
    }

    // 将结果写回全局内存 C
    if (global_row < M && global_col < N) {
        C[global_row * N + global_col] = Cvalue;
    }
}

// 矩阵转置乘法: 分块优化 C = A * B^T
template<typename T>
__global__ void matmul_transposed_tiled_A_B_T_kernel(const T* A, const T* B, T* C, int M, int N, int K) {
    const int TILE_SIZE = ZEROLLM_DEFAULT_TILE_SIZE;
    
    // 使用静态声明的共享内存
    __shared__ T As[TILE_SIZE][TILE_SIZE + 1];  // +1 避免 bank 冲突
    __shared__ T Bs[TILE_SIZE][TILE_SIZE + 1];

    // 线程在块内的坐标
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    // C 矩阵的全局坐标
    int global_row = blockIdx.y * TILE_SIZE + ty;
    int global_col = blockIdx.x * TILE_SIZE + tx;

    T Cvalue = 0;

    // 遍历 K 维度的瓦片
    for (int t = 0; t < K; t += TILE_SIZE) {
        // 载入 A 的子矩阵 (A[global_row, t:t+TILE_SIZE])
        if (global_row < M && (t + tx) < K)
            As[ty][tx] = A[global_row * K + t + tx];
        else
            As[ty][tx] = 0;

        // 载入 B^T 的子矩阵 (B^T[t:t+TILE_SIZE, global_col])
        // 这实际上是 B[global_col, t:t+TILE_SIZE]
        if ((t + ty) < K && global_col < N)
            Bs[ty][tx] = B[global_col * K + t + ty];
        else
            Bs[ty][tx] = 0;

        __syncthreads();

        // 在共享内存上计算子矩阵乘积
        for (int k = 0; k < TILE_SIZE; ++k) {
            Cvalue += As[ty][k] * Bs[k][tx];  // 注意这里B是转置的
        }

        __syncthreads();
    }

    // 将结果写回全局内存 C
    if (global_row < M && global_col < N) {
        C[global_row * N + global_col] = Cvalue;
    }
}


template<typename T>
__global__ void matmul_transposed_tiled_A_T_B_kernel(const T* A, const T* B, T* C, int M, int N, int K) {
    // 目标: C[K, N] = A^T[K, M] * B[M, N]
    // A 物理形状: [M, K]
    // B 物理形状: [M, N]
    // C 物理形状: [K, N]
    // M = 归约维度 (Reduction Dim)
    // N = C 的列数 (C_cols)
    // K = C 的行数 (C_rows)

    const int TILE_SIZE = ZEROLLM_DEFAULT_TILE_SIZE;
    
    __shared__ T As[TILE_SIZE][TILE_SIZE + 1];
    __shared__ T Bs[TILE_SIZE][TILE_SIZE + 1];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    // global_row 对应 C 的行 (0..K-1)
    int global_row = blockIdx.y * TILE_SIZE + ty;
    // global_col 对应 C 的列 (0..N-1)
    int global_col = blockIdx.x * TILE_SIZE + tx; 

    T Cvalue = 0;

    // *** 修正 1: 必须遍历累加维度 M ***
    for (int t = 0; t < M; t += TILE_SIZE) {
        
        // 加载 A 的块: A^T[global_row, t+k] => A[t+k, global_row]
        // As[ty][tx] 用于 As[ty][k]
        // As[k][tx] 用于 As[k][tx]
        // 我们需要 As[k][ty] = A[t+k, global_row]
        // 我们需要 Bs[k][tx] = B[t+k, global_col]
        // 这与 C = A*B (As[ty][k], Bs[k][tx]) 的加载模式不同

        // 简化的加载: As[ty][tx] = A^T[global_row+ty, t+tx] = A[t+tx, global_row+ty]
        // Bs[ty][tx] = B[t+ty, global_col+tx]
        // 不，我们坚持 C[row, col] += As[row, k] * Bs[k, col] 的模式

        // As[ty][tx] 加载 A^T 瓦片: A^T[global_row, t+tx] => A[t+tx, global_row]
        // *** 修正 2 & 3: 使用 M 检查, 使用 K 步幅 ***
        if ((t + tx) < M && global_row < K)
            As[ty][tx] = A[(t + tx) * K + global_row]; 
        else
            As[ty][tx] = 0;

        // Bs[ty][tx] 加载 B 瓦片: B[t+ty, global_col]
        // *** 修正 3: 使用 M 检查 ***
        if ((t + ty) < M && global_col < N)
            Bs[ty][tx] = B[(t + ty) * N + global_col];
        else
            Bs[ty][tx] = 0;

        __syncthreads();

        // 计算 C_sub[ty, tx] += As[ty, k] * Bs[k, tx]
        // As[ty][k] = A^T[global_row, t+k] = A[t+k, global_row]
        // Bs[k][tx] = B[t+k, global_col]
        // 这是正确的计算
        for (int k = 0; k < TILE_SIZE; ++k) {
            Cvalue += As[ty][k] * Bs[k][tx]; 
        }

        __syncthreads();
    }

    if (global_row < K && global_col < N) {
        C[global_row * N + global_col] = Cvalue;
    }
}
// 矩阵乘法: C = A * B
template<typename T>
void matmul(const T* A, const T* B, T* C, int M, int N, int K, cudaStream_t stream) {
    const int BLOCK_DIM = static_cast<int>(std::sqrt(ZEROLLM_DEFAULT_THREADS));
    if (BLOCK_DIM <= 0) {
        throw std::runtime_error("ZEROLLM_DEFAULT_THREADS too small for sqrt.");
    }

    dim3 block(BLOCK_DIM, BLOCK_DIM);
    dim3 grid((N + BLOCK_DIM - 1) / BLOCK_DIM, (M + BLOCK_DIM - 1) / BLOCK_DIM);

    matmul_kernel<<<grid, block, 0, stream>>>(A, B, C, M, N, K);
    CHECK(cudaGetLastError(), "matmul_kernel launch failed");
}

// 矩阵转置乘法: C = A^T * B
template<typename T>
void matmul_transposed_A_T_B(const T* A, const T* B, T* C, int M, int N, int K, cudaStream_t stream) {
    const int BLOCK_DIM = static_cast<int>(std::sqrt(ZEROLLM_DEFAULT_THREADS));
    if (BLOCK_DIM <= 0) {
        throw std::runtime_error("ZEROLLM_DEFAULT_THREADS too small for sqrt.");
    }

    dim3 block(BLOCK_DIM, BLOCK_DIM);
    dim3 grid((N + BLOCK_DIM - 1) / BLOCK_DIM, (K + BLOCK_DIM - 1) / BLOCK_DIM);

    matmul_transposed_A_T_B_kernel<<<grid, block, 0, stream>>>(A, B, C, M, N, K);
    CHECK(cudaGetLastError(), "matmul_transposed_A_T_B_kernel launch failed");
}

// 矩阵转置乘法: C = A * B^T
template<typename T>
void matmul_transposed_A_B_T(const T* A, const T* B, T* C, int M, int N, int K, cudaStream_t stream) {
    const int BLOCK_DIM = static_cast<int>(std::sqrt(ZEROLLM_DEFAULT_THREADS));
    if (BLOCK_DIM <= 0) {
        throw std::runtime_error("ZEROLLM_DEFAULT_THREADS too small for sqrt.");
    }

    dim3 block(BLOCK_DIM, BLOCK_DIM);
    dim3 grid((N + BLOCK_DIM - 1) / BLOCK_DIM, (M + BLOCK_DIM - 1) / BLOCK_DIM);

    matmul_transposed_A_B_T_kernel<<<grid, block, 0, stream>>>(A, B, C, M, N, K);
    CHECK(cudaGetLastError(), "matmul_transposed_A_B_T_kernel launch failed");
}

// 矩阵乘法: 分块优化
template<typename T>
void matmul_tiled(const T* A, const T* B, T* C, int M, int N, int K, cudaStream_t stream) {
    const int TILE_SIZE = ZEROLLM_DEFAULT_TILE_SIZE;

    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);
    size_t shmem_size = 2 * TILE_SIZE * (TILE_SIZE + 1) * sizeof(T);

    matmul_tiled_kernel<<<grid, block, shmem_size, stream>>>(A, B, C, M, N, K);
    CHECK(cudaGetLastError(), "matmul_tiled_kernel launch failed");
}

template<typename T>
void matmul_transposed_tiled_A_B_T(const T* A, const T* B, T* C, int M, int N, int K, cudaStream_t stream) {
    const int TILE_SIZE = ZEROLLM_DEFAULT_TILE_SIZE;

    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);
    size_t shmem_size = 2 * TILE_SIZE * (TILE_SIZE + 1) * sizeof(T);

    matmul_transposed_tiled_A_B_T_kernel<<<grid, block, shmem_size, stream>>>(A, B, C, M, N, K);
    CHECK(cudaGetLastError(), "matmul_transposed_tiled_A_B_T_kernel launch failed");
}

template<typename T>
void matmul_transposed_tiled_A_T_B(const T* A, const T* B, T* C, int M, int N, int K, cudaStream_t stream) {
    const int TILE_SIZE = ZEROLLM_DEFAULT_TILE_SIZE;

    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE, (K + TILE_SIZE - 1) / TILE_SIZE);
    size_t shmem_size = 2 * TILE_SIZE * (TILE_SIZE + 1) * sizeof(T);

    matmul_transposed_tiled_A_T_B_kernel<<<grid, block, shmem_size, stream>>>(A, B, C, M, N, K);
    CHECK(cudaGetLastError(), "matmul_transposed_tiled_A_T_B_kernel launch failed");
}

// 显式实例化
template void matmul<float>(const float* A, const float* B, float* C, int M, int N, int K, cudaStream_t stream);
template void matmul_transposed_A_T_B<float>(const float* A, const float* B, float* C, int M, int N, int K, cudaStream_t stream);
template void matmul_transposed_A_B_T<float>(const float* A, const float* B, float* C, int M, int N, int K, cudaStream_t stream);
template void matmul_tiled<float>(const float* A, const float* B, float* C, int M, int N, int K, cudaStream_t stream);
template void matmul_transposed_tiled_A_B_T<float>(const float* A, const float* B, float* C, int M, int N, int K, cudaStream_t stream);
template void matmul_transposed_tiled_A_T_B<float>(const float* A, const float* B, float* C, int M, int N, int K, cudaStream_t stream);
