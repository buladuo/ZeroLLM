#ifndef TRANSPOSE_CUH
#define TRANSPOSE_CUH

#include <cuda_runtime.h>
#include <cuda.h>

/**
 * @brief 矩阵转置
 * 
 * @tparam T 数据类型
 * @param input 输入矩阵 [rows, cols]
 * @param output 输出矩阵 [cols, rows]
 * @param rows 输入矩阵行数
 * @param cols 输入矩阵列数
 * @param stream 运行时流
 */
template<typename T>
void transpose(const T* input, T* output, int rows, int cols, 
               cudaStream_t stream = 0);


#endif // TRANSPOSE_CUH