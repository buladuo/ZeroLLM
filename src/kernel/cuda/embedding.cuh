#ifndef EMBEDDING_CUH
#define EMBEDDING_CUH

#include <cuda_runtime.h>
#include <cuda.h>

/**
 * @brief Embedding前向传播
 * 
 * @tparam T 数据类型
 * @tparam IdxType 索引类型
 * @param input 输入索引, shape [batch_size, seq_len]
 * @param embedding_table 嵌入表, shape [vocab_size, embed_dim]
 * @param pos_encoding_table 位置编码表, shape [max_seq_len, embed_dim]
 * @param output 输出嵌入向量, shape [batch_size, seq_len, embed_dim]
 * @param batch_size 批处理大小
 * @param seq_len 序列长度
 * @param embed_dim 嵌入维度
 * @param vocab_size 词汇表大小
 * @param max_seq_len 最大序列长度
 * @param stream CUDA流
 */
template<typename T, typename IdxType>
void cuda_embedding_forward(const IdxType* input,
                           const T* embedding_table,
                           const T* pos_encoding_table,
                           T* output,
                           int batch_size,
                           int seq_len,
                           int embed_dim,
                           int vocab_size,
                           int max_seq_len,
                           cudaStream_t stream = 0);

/**
 * @brief Embedding反向传播
 * 
 * @tparam T 数据类型
 * @tparam IdxType 索引类型
 * @param input 输入索引, shape [batch_size, seq_len]
 * @param d_output 输出梯度, shape [batch_size, seq_len, embed_dim]
 * @param d_embedding_table 嵌入表梯度, shape [vocab_size, embed_dim]
 * @param batch_size 批处理大小
 * @param seq_len 序列长度
 * @param embed_dim 嵌入维度
 * @param vocab_size 词汇表大小
 * @param stream CUDA流
 */
template<typename T, typename IdxType>
void cuda_embedding_backward(const IdxType* input,
                            const T* d_output,
                            T* d_embedding_table,
                            int batch_size,
                            int seq_len,
                            int embed_dim,
                            int vocab_size,
                            cudaStream_t stream = 0);

#endif // EMBEDDING_CUH