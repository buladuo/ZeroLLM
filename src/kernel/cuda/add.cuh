#ifndef ADD_CUH
#define ADD_CUH

#include <cuda_runtime.h>
#include <cuda.h>

/**
 * @brief 向量/矩阵逐元素相加: output = a + b
 * 
 * @tparam T 数据类型
 * @param a 输入向量a
 * @param b 输入向量b
 * @param output 输出向量
 * @param size 向量元素总数 (例如 M*N)
 * @param stream CUDA流
 */
template<typename T>
void add(const T* a, const T* b, T* output, int size, cudaStream_t stream = 0);


/**
 * @brief 向量/矩阵逐元素相加 (in-place): a = a + b
 * 
 * @tparam T 数据类型
 * @param a 输入输出向量a（in-place操作）
 * @param b 输入向量b
 * @param size 向量元素总数
 * @param stream CUDA流
 */
template<typename T>
void add_inplace(T* a, const T* b, int size, cudaStream_t stream = 0);

#endif // ADD_CUH
