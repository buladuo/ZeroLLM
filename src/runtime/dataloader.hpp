#ifndef DATALOADER_HPP
#define DATALOADER_HPP

#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
#include <stdexcept>
#include <thread> 
#include <future> 

// 结构定义保持不变
struct BatchData {
    std::vector<size_t> inputs;
    std::vector<size_t> targets;
    size_t batch_size;
    size_t seq_len;
};

struct SampleIndex {
    size_t story_idx;
    size_t start_token_idx;
};

class LLMDataLoader {
public:
    LLMDataLoader(const std::vector<std::vector<size_t>>& token_lists,
                  size_t batch_size,
                  size_t seq_len,
                  size_t eos_id,
                  size_t pad_id);

    void reset();
    bool has_next() const;
    BatchData next_batch();
    size_t total_batches() const;

private:
    const std::vector<std::vector<size_t>>* data_ptr_;
    std::vector<SampleIndex> sample_indices_;
    
    size_t batch_size_;
    size_t seq_len_;
    size_t eos_id_;
    size_t pad_id_;
    
    size_t current_idx_ = 0;
    std::mt19937 generator_;

    void create_sample_indices_parallel(); 
    
    size_t get_token_at(size_t story_idx, size_t logical_pos) const;
};
#endif // DATALOADER_HPP