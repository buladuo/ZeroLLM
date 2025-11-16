#include "embedding.cuh"
#include <iostream>
#include "config.hpp"

template<typename T, typename IdxType>
__global__ void embedding_forward_kernel(const IdxType* input,
                                       const T* embedding_table,
                                       const T* pos_encoding_table,
                                       T* output,
                                       int batch_size,
                                       int seq_len,
                                       int embed_dim,
                                       int vocab_size,
                                       int max_seq_len) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * seq_len * embed_dim;
    
    if (idx >= total_elements) return;
    
    int i = idx / (seq_len * embed_dim);  // batch index
    int j = (idx / embed_dim) % seq_len;  // sequence index
    int k = idx % embed_dim;              // embedding dimension index
    
    if (j >= max_seq_len) return;
    
    IdxType token_id = input[i * seq_len + j];
    if (token_id < 0 || token_id >= vocab_size) return;
    
    output[idx] = embedding_table[token_id * embed_dim + k] + pos_encoding_table[j * embed_dim + k];
}

template<typename T, typename IdxType>
__global__ void embedding_backward_kernel(const IdxType* input,
                                        const T* d_output,
                                        T* d_embedding_table,
                                        int batch_size,
                                        int seq_len,
                                        int embed_dim,
                                        int vocab_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * seq_len * embed_dim;
    
    if (idx >= total_elements) return;
    
    int i = idx / (seq_len * embed_dim);  // batch index
    int j = (idx / embed_dim) % seq_len;  // sequence index
    int k = idx % embed_dim;              // embedding dimension index
    
    IdxType token_id = input[i * seq_len + j];
    if (token_id < 0 || token_id >= vocab_size) return;
    
    atomicAdd(&d_embedding_table[token_id * embed_dim + k], d_output[idx]);
}

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
                           cudaStream_t stream) {
    if (batch_size <= 0 || seq_len <= 0 || embed_dim <= 0) return;
    
    int total_elements = batch_size * seq_len * embed_dim;
    int block_size = ZEROLLM_DEFAULT_THREADS;
    int grid_size = ZEROLLM_CALC_BLOCKS(total_elements);
    
    embedding_forward_kernel<<<grid_size, block_size, 0, stream>>>(
        input, embedding_table, pos_encoding_table, output,
        batch_size, seq_len, embed_dim, vocab_size, max_seq_len);
    CHECK(cudaGetLastError(), "embedding_forward_kernel launch failed");
}

template<typename T, typename IdxType>
void cuda_embedding_backward(const IdxType* input,
                            const T* d_output,
                            T* d_embedding_table,
                            int batch_size,
                            int seq_len,
                            int embed_dim,
                            int vocab_size,
                            cudaStream_t stream) {
    if (batch_size <= 0 || seq_len <= 0 || embed_dim <= 0) return;
    
    int total_elements = batch_size * seq_len * embed_dim;
    int block_size = ZEROLLM_DEFAULT_THREADS;
    int grid_size = ZEROLLM_CALC_BLOCKS(total_elements);
    
    embedding_backward_kernel<<<grid_size, block_size, 0, stream>>>(
        input, d_output, d_embedding_table,
        batch_size, seq_len, embed_dim, vocab_size);
    CHECK(cudaGetLastError(), "embedding_backward_kernel launch failed");
}

// 显式实例化
template void cuda_embedding_forward<float, int>(const int* input,
                                                const float* embedding_table,
                                                const float* pos_encoding_table,
                                                float* output,
                                                int batch_size,
                                                int seq_len,
                                                int embed_dim,
                                                int vocab_size,
                                                int max_seq_len,
                                                cudaStream_t stream);

template void cuda_embedding_backward<float, int>(const int* input,
                                                 const float* d_output,
                                                 float* d_embedding_table,
                                                 int batch_size,
                                                 int seq_len,
                                                 int embed_dim,
                                                 int vocab_size,
                                                 cudaStream_t stream);