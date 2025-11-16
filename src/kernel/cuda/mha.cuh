#ifndef MHA_CUH
#define MHA_CUH

#include <cuda_runtime.h>
#include <cuda.h>
#include <stdint.h>

/**
 * @brief MHA 前向核/接口
 *
 * Q, K, V layout: flattened as [batch * seq_len * embed_dim]
 * For head accesses: head offset = head_idx * head_dim inside embed_dim.
 *
 * attention_scores should have shape: [batch * num_heads * seq_len * seq_len]
 * attn_output has same layout as Q/K/V input: [batch * seq_len * embed_dim]
 *
 */
template<typename T>
void mha_forward(const T* Q, const T* K, const T* V,
                      T* attn_output, T* attention_scores,
                      const bool* mask,
                      int64_t batch_size, int64_t seq_len,
                      int64_t num_heads, int64_t head_dim,
                      bool is_causal,
                      cudaStream_t stream = 0);


/**
 * @brief MHA 反向核/接口
 *
 * 输入:
 *   d_attn_output: [batch * seq_len * embed_dim]
 *   Q,K,V: 原始前向输入
 *   attention_scores: 前向 softmax 输出 (probabilities)
 * 输出:
 *   d_Q, d_K, d_V: gradient buffers (same layout as Q/K/V)
 *
 */
template<typename T>
void mha_backward(const T* d_attn_output,
                       const T* Q, const T* K, const T* V,
                       const T* attention_scores,
                       const bool* mask,
                       T* d_Q, T* d_K, T* d_V,
                       int64_t batch_size, int64_t seq_len,
                       int64_t num_heads, int64_t head_dim,
                       bool is_causal,
                       cudaStream_t stream = 0);

#endif // MHA_CUH
