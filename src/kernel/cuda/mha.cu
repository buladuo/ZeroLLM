#include "mha.cuh"
#include "config.hpp"
#include <cmath>
#include <float.h>
#include <algorithm>
#include <stdexcept>

//
// 说明：此实现以“清晰正确”为优先，逐元素计算注意力分数、softmax、以及 backward。
//       对于中等 seq_len/heads 这是易于理解且正确的实现；如需性能可继续优化。
//


// 每个 block 处理一个 (batch, head) 对应的全部计算。
// blockIdx.x => b * num_heads + head
// 使用单一线程块内的线程做行/列循环拆分。
template<typename T>
__global__ void mha_forward_block_kernel(const T* __restrict__ Q,
                                         const T* __restrict__ K,
                                         const T* __restrict__ V,
                                         T* __restrict__ attn_output,
                                         T* __restrict__ attention_scores,
                                         const bool* __restrict__ mask,
                                         int64_t batch_size,
                                         int64_t seq_len,
                                         int64_t embed_dim,
                                         int64_t num_heads,
                                         int64_t head_dim,
                                         bool is_causal) {
    int block_idx = blockIdx.x;
    int b = block_idx / num_heads;
    int head = block_idx % num_heads;

    if (b >= batch_size) return;

    // stride offsets
    int64_t batch_seq_offset = (int64_t)b * seq_len;
    int64_t embed_head_offset = (int64_t)head * head_dim; // inside embed_dim

    // pointers base
    // Q/K/V are [batch * seq_len * embed_dim]
    const T* Q_base = Q + ((int64_t)b * seq_len * embed_dim) + embed_head_offset;
    const T* K_base = K + ((int64_t)b * seq_len * embed_dim) + embed_head_offset;
    const T* V_base = V + ((int64_t)b * seq_len * embed_dim) + embed_head_offset;

    // attention_scores layout for this (b, head) block:
    // base index = ((b * num_heads + head) * seq_len * seq_len)
    T* scores_base = attention_scores + ((int64_t)(b * num_heads + head) * seq_len * seq_len);

    // scaling factor 1/sqrt(d_k)
    const T scale = T(1.0f / sqrtf((float)head_dim));

    // thread index for parallelizing inner loops
    int tid = threadIdx.x;
    int nthreads = blockDim.x;

    // For each query row i compute scores to all keys j, then softmax, then weighted sum over V.
    for (int i = tid; i < seq_len; i += nthreads) {
        // 1) compute raw scores for row i: score[j] = dot(Q_i, K_j) * scale
        // We compute max for numerical stability
        T row_max = -FLT_MAX;
        // temporary buffer for scores — write directly to global scores_base
        for (int j = 0; j < seq_len; ++j) {
            // if causal and j > i -> masked
            if (is_causal && j > i) {
                scores_base[i * seq_len + j] = -FLT_MAX;
                continue;
            }
            if (mask != nullptr) {
                // mask is seq_len * seq_len on host/gpu: true means keep, false means mask
                bool keep = mask[i * seq_len + j];
                if (!keep) {
                    scores_base[i * seq_len + j] = -FLT_MAX;
                    continue;
                }
            }
            // dot product over head_dim
            float dot = 0.0f;
            for (int d = 0; d < head_dim; ++d) {
                float qv = Q_base[i * embed_dim + d]; // careful indexing: embed_dim stride
                float kv = K_base[j * embed_dim + d];
                dot += qv * kv;
            }
            float s = dot * scale;
            scores_base[i * seq_len + j] = s;
            if (s > row_max) row_max = s;
        }

        // 2) compute exp(x - max) and sum
        float row_sum = 0.0f;
        for (int j = 0; j < seq_len; ++j) {
            float s = scores_base[i * seq_len + j];
            if (s == -FLT_MAX) { // masked
                scores_base[i * seq_len + j] = 0.0f;
            } else {
                float e = expf(s - row_max);
                scores_base[i * seq_len + j] = e;
                row_sum += e;
            }
        }

        // 3) normalize -> store softmax probabilities back into scores_base
        if (row_sum == 0.0f) row_sum = 1e-20f; // avoid div0
        for (int j = 0; j < seq_len; ++j) {
            scores_base[i * seq_len + j] = scores_base[i * seq_len + j] / row_sum;
        }

        // 4) compute output vector for this query i: out_i_head[d] = sum_j prob(i,j) * V_j[d]
        for (int d = 0; d < head_dim; ++d) {
            float out_val = 0.0f;
            for (int j = 0; j < seq_len; ++j) {
                float p = scores_base[i * seq_len + j];
                if (p == 0.0f) continue;
                float v = V_base[j * embed_dim + d];
                out_val += p * v;
            }
            // write to attn_output at correct embed location:
            // attn_output offset: (b * seq_len + i) * embed_dim + head * head_dim + d
            attn_output[((int64_t)b * seq_len + i) * embed_dim + embed_head_offset + d] = out_val;
        }
    }
}


template<typename T>
void mha_forward(const T* Q, const T* K, const T* V,
                      T* attn_output, T* attention_scores,
                      const bool* mask,
                      int64_t batch_size, int64_t seq_len,
                      int64_t num_heads, int64_t head_dim,
                      bool is_causal,
                      cudaStream_t stream) {
    if (batch_size <= 0 || seq_len <= 0 || num_heads <= 0 || head_dim <= 0) {
        throw std::invalid_argument("mha_forward: invalid sizes");
    }

    // embed_dim = num_heads * head_dim
    int64_t embed_dim = num_heads * head_dim;

    // 每个 block 处理一个 (batch, head)
    int blocks = static_cast<int>(batch_size * num_heads);
    int threads = ZEROLLM_DEFAULT_THREADS;
    if (threads <= 0) threads = 128;

    mha_forward_block_kernel<<<blocks, threads, 0, stream>>>(Q, K, V, attn_output, attention_scores, mask,
                                                             batch_size, seq_len, embed_dim, num_heads, head_dim, is_causal);
    CHECK(cudaGetLastError(), "mha_forward_block_kernel launch failed");
}


// ------------------ backward ------------------
// Backward kernels follow straightforward matrix calculus:
// Given: out_b,i,h,d = sum_j softmax(i,j) * V_b,j,h,d
// dV: accumulate p(i,j) * d_out(i)  (sum over queries i)
// d_soft = d_out(i) dot V_j  -> yields scalar per (i,j) per head and d (but we sum over d to get scalar per (i,j))
// d_scores = softmax_backward(softmax_ij, d_soft_ij)
// dQ += sum_j d_scores(i,j) * K_j
// dK += sum_i d_scores(i,j) * Q_i
//
// Here we implement these steps in straightforward loops.
//

template<typename T>
__global__ void mha_backward_block_kernel(const T* __restrict__ d_attn_output,
                                          const T* __restrict__ Q,
                                          const T* __restrict__ K,
                                          const T* __restrict__ V,
                                          const T* __restrict__ attention_scores, // softmax probs
                                          const bool* __restrict__ mask,
                                          T* __restrict__ d_Q,
                                          T* __restrict__ d_K,
                                          T* __restrict__ d_V,
                                          int64_t batch_size,
                                          int64_t seq_len,
                                          int64_t embed_dim,
                                          int64_t num_heads,
                                          int64_t head_dim,
                                          bool is_causal) {
    int block_idx = blockIdx.x;
    int b = block_idx / num_heads;
    int head = block_idx % num_heads;
    if (b >= batch_size) return;

    int64_t embed_head_offset = (int64_t)head * head_dim;
    const T* Q_base = Q + ((int64_t)b * seq_len * embed_dim) + embed_head_offset;
    const T* K_base = K + ((int64_t)b * seq_len * embed_dim) + embed_head_offset;
    const T* V_base = V + ((int64_t)b * seq_len * embed_dim) + embed_head_offset;

    const T* d_out_base = d_attn_output + ((int64_t)b * seq_len * embed_dim) + embed_head_offset;

    T* d_Q_base = d_Q + ((int64_t)b * seq_len * embed_dim) + embed_head_offset;
    T* d_K_base = d_K + ((int64_t)b * seq_len * embed_dim) + embed_head_offset;
    T* d_V_base = d_V + ((int64_t)b * seq_len * embed_dim) + embed_head_offset;

    const T* probs_base = attention_scores + ((int64_t)(b * num_heads + head) * seq_len * seq_len);

    int tid = threadIdx.x;
    int nthreads = blockDim.x;

    // initialize gradients to zero for this (b,head) region (thread-parallel)
    for (int idx = tid; idx < (int)(seq_len * head_dim); idx += nthreads) {
        int pos = idx;
        int token = pos / head_dim;
        int d = pos % head_dim;
        d_Q_base[token * embed_dim + d] = 0.0f;
        d_K_base[token * embed_dim + d] = 0.0f;
        d_V_base[token * embed_dim + d] = 0.0f;
    }
    __syncthreads();

    // 1) compute d_V:
    // d_V_j_d = sum_i p(i,j) * d_out_i_d
    for (int j = tid; j < seq_len; j += nthreads) {
        for (int d = 0; d < head_dim; ++d) {
            float acc = 0.0f;
            for (int i = 0; i < seq_len; ++i) {
                // mask check
                if (is_causal && j > i) continue;
                if (mask != nullptr) {
                    if (!mask[i * seq_len + j]) continue;
                }
                float p = probs_base[i * seq_len + j];
                if (p == 0.0f) continue;
                float dout = d_out_base[i * embed_dim + d];
                acc += p * dout;
            }
            // accumulate to d_V_base[j, d]
            // Note: multiple threads write to distinct j here, so safe.
            d_V_base[j * embed_dim + d] = acc;
        }
    }
    __syncthreads();

    // 2) compute d_soft = for each (i,j): sum_d d_out_i_d * V_j_d
    // and then d_scores = softmax_backward: d_scores(i,j) = sum_d d_soft(i,j)_d ... (we compute scalar)
    // Finally accumulate into dQ and dK via matmul with d_scores
    // We'll compute per i (rows) in parallel by threads
    for (int i = tid; i < seq_len; i += nthreads) {
        // compute d_soft (scalar per j)
        // d_soft_j = dot(d_out_i (head_dim), V_j (head_dim))
        // then d_scores_j = softmax_back_scalar( p_ij, d_soft_j )
        // where softmax_back_scalar: given probs p_k and upstream g_k, gradient for input score s_j is:
        //   d_s_j = sum_k (Jacobian_{j,k} * g_k) = p_j * (g_j - sum_k p_k g_k)
        // here g_k is d_soft_k
        // but we treat g_k as d_soft(i,k)
        float sum_p_g = 0.0f;
        // first compute g_k and p_k and sum_p_g
        // store g_k temporary in shared global memory? we'll use local vector
        extern __shared__ float tmp[]; // not used in launch; but safe to declare
        // allocate on heap-like local; but seq_len unknown - so recompute twice instead
        // compute sum_p_g
        for (int j = 0; j < seq_len; ++j) {
            if (is_causal && j > i) continue;
            if (mask != nullptr) {
                if (!mask[i * seq_len + j]) continue;
            }
            float p = probs_base[i * seq_len + j];
            if (p == 0.0f) continue;
            // compute g_j = dot(d_out_i, V_j)
            float g = 0.0f;
            for (int d = 0; d < head_dim; ++d) {
                float dout = d_out_base[i * embed_dim + d];
                float v = V_base[j * embed_dim + d];
                g += dout * v;
            }
            sum_p_g += p * g;
        }

        // now compute d_scores_j and accumulate into dQ and dK
        for (int j = 0; j < seq_len; ++j) {
            if (is_causal && j > i) continue;
            if (mask != nullptr) {
                if (!mask[i * seq_len + j]) continue;
            }
            float p = probs_base[i * seq_len + j];
            if (p == 0.0f) continue;
            // recompute g_j
            float g = 0.0f;
            for (int d = 0; d < head_dim; ++d) {
                float dout = d_out_base[i * embed_dim + d];
                float v = V_base[j * embed_dim + d];
                g += dout * v;
            }
            float d_score = p * (g - sum_p_g); // gradient wrt score s_ij
            // accumulate into dQ_i += d_score * K_j
            for (int d = 0; d < head_dim; ++d) {
                float k = K_base[j * embed_dim + d];
                // atomic add? different threads may update same d_Q_base entry when i assigned to multiple threads?
                // Here each i is handled by single thread index (tid loop), so only this thread updates d_Q_base for token i
                // So safe to direct add.
                atomicAdd(&d_Q_base[i * embed_dim + d], d_score * k);
                // accumulate into d_K_j += d_score * Q_i  -> multiple i may update same j, so use atomicAdd
                float q = Q_base[i * embed_dim + d];
                atomicAdd(&d_K_base[j * embed_dim + d], d_score * q);
            }
        }
    }
    __syncthreads();
}


template<typename T>
void mha_backward(const T* d_attn_output,
                       const T* Q, const T* K, const T* V,
                       const T* attention_scores,
                       const bool* mask,
                       T* d_Q, T* d_K, T* d_V,
                       int64_t batch_size, int64_t seq_len,
                       int64_t num_heads, int64_t head_dim,
                       bool is_causal,
                       cudaStream_t stream) {
    if (batch_size <= 0 || seq_len <= 0 || num_heads <= 0 || head_dim <= 0) {
        throw std::invalid_argument("cuda_mha_backward: invalid sizes");
    }

    int64_t embed_dim = num_heads * head_dim;
    int blocks = static_cast<int>(batch_size * num_heads);
    int threads = ZEROLLM_DEFAULT_THREADS;
    if (threads <= 0) threads = 128;

    // Note: backward kernel uses atomicAdd for accumulating d_K across threads handling different i's.
    // Launch kernel with some shared memory (we don't rely on it currently).
    size_t shmem = 0;
    mha_backward_block_kernel<<<blocks, threads, shmem, stream>>>(d_attn_output, Q, K, V, attention_scores, mask,
                                                                  d_Q, d_K, d_V,
                                                                  batch_size, seq_len, embed_dim, num_heads, head_dim, is_causal);
    CHECK(cudaGetLastError(), "mha_backward_block_kernel launch failed");
}


// 显式实例化 float
template void mha_forward<float>(const float* Q, const float* K, const float* V,
                                        float* attn_output, float* attention_scores,
                                        const bool* mask,
                                        int64_t batch_size, int64_t seq_len,
                                        int64_t num_heads, int64_t head_dim,
                                        bool is_causal,
                                        cudaStream_t stream);

template void mha_backward<float>(const float* d_attn_output,
                                        const float* Q, const float* K, const float* V,
                                        const float* attention_scores,
                                        const bool* mask,
                                        float* d_Q, float* d_K, float* d_V,
                                        int64_t batch_size, int64_t seq_len,
                                        int64_t num_heads, int64_t head_dim,
                                        bool is_causal,
                                        cudaStream_t stream);