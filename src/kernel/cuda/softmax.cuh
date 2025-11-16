#ifndef SOFTMAX_CUH
#define SOFTMAX_CUH

#include <cuda_runtime.h>
#include <cuda.h>

/**
 * @brief 逐行Softmax计算
 * 
 * @tparam T 数据类型
 * @param input 输入矩阵 [M, N]
 * @param output 输出矩阵 [M, N]
 * @param M 行数（批次大小）
 * @param N 列数（每个样本的特征维度）
 * @param stream CUDA流
 */
template<typename T>
void softmax(const T* input, T* output, int M, int N, cudaStream_t stream = 0);

#endif // SOFTMAX_CUH
