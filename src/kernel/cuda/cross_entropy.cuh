#pragma once
#include <cuda_runtime.h>

/**
 * @brief Cross Entropy Loss CUDA内核函数
 */

/**
 * @brief Cross Entropy Loss反向传播Kernel
 * @param d_logits logits梯度输出 [batch_size, num_classes]
 * @param softmax_output softmax输出 [batch_size, num_classes]
 * @param targets 真实标签 [batch_size]
 * @param batch_size 批次大小
 * @param num_classes 类别数量
 */
void cross_entropy_backward_kernel(float* d_logits,
                                   const float* softmax_output,
                                   const int* targets,
                                   int batch_size,
                                   int num_classes,
                                   cudaStream_t stream = 0);