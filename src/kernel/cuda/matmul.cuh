#ifndef MATMUL_CUH
#define MATMUL_CUH

#include <cuda_runtime.h>
#include <cuda.h>

/**
 * @brief 矩阵乘法: C = A * B
 * 
 * @tparam T 数据类型
 * @param A 矩阵A, shape [M, K]
 * @param B 矩阵B, shape [K, N]
 * @param C 输出矩阵C, shape [M, N]
 * @param M A的行数
 * @param N B的列数
 * @param K A的列数/B的行数
 * @param stream CUDA流
 */
template<typename T>
void matmul(const T* A, const T* B, T* C, int M, int N, int K, cudaStream_t stream = 0);


/**
 * @brief 矩阵转置乘法: C = A^T * B
 * 
 * @tparam T 数据类型
 * @param A 矩阵A, shape [M, K]
 * @param B 矩阵B, shape [K, N]
 * @param C 输出矩阵C, shape [K, N]
 * @param M A的行数
 * @param N B的列数
 * @param K A的列数/B的行数
 * @param stream CUDA流
 */
template<typename T>
void matmul_transposed_A_T_B(const T* A, const T* B, T* C, int M, int N, int K, cudaStream_t stream = 0);


/**
 * @brief 矩阵转置乘法: C = A * B^T
 * 
 * @tparam T 数据类型
 * @param A 矩阵A, shape [M, K]
 * @param B 矩阵B, shape [N, K]
 * @param C 输出矩阵C, shape [M, N]
 * @param M A的行数
 * @param N B的行数
 * @param K A的列数/B的列数
 * @param stream CUDA流        
 */
template<typename T>
void matmul_transposed_A_B_T(const T* A, const T* B, T* C, int M, int N, int K, cudaStream_t stream = 0);


/**
 * @brief 矩阵乘法：使用分块优化的矩阵乘法 C = A * B
 * 
 * @tparam T 数据类型
 * @param A 矩阵A, shape [M, K]
 * @param B 矩阵B, shape [K, N]
 * @param C 输出矩阵C, shape [M, N]
 * @param M A的行数
 * @param N B的列数
 * @param K A的列数/B的行数
 * @param stream CUDA流
 */
template<typename T>            
void matmul_tiled(const T* A, const T* B, T* C, int M, int N, int K, cudaStream_t stream = 0);

/**
 * @brief 矩阵转置乘法：使用分块优化的矩阵乘法 C = A * B^T
 * 
 * @tparam T 数据类型
 * @param A 矩阵A, shape [M, K]
 * @param B 矩阵B, shape [N, K]
 * @param C 输出矩阵C, shape [M, N]
 * @param M A的行数
 * @param N B的行数
 * @param K A的列数/B的列数
 * @param stream CUDA流        
 */           
template<typename T>
void matmul_transposed_tiled_A_B_T(const T* A, const T* B, T* C, int M, int N, int K, cudaStream_t stream = 0);


/**
 * @brief 矩阵转置乘法：使用分块优化的矩阵乘法 C = A^T * B
 * 
 * @tparam T 数据类型
 * @param A 矩阵A, shape [M, K]
 * @param B 矩阵B, shape [K, N]
 * @param C 输出矩阵C, shape [K, N]
 * @param M A的行数
 * @param N B的列数
 * @param K A的列数/B的行数
 * @param stream CUDA流        
 */ 
template<typename T>
void matmul_transposed_tiled_A_T_B(const T* A, const T* B, T* C, int M, int N, int K, cudaStream_t stream = 0);

#endif // MATMUL_CUH
