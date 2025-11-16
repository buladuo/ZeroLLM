#include "dataloader.hpp"
#include <stdexcept>
#include <cmath>

LLMDataLoader::LLMDataLoader(const std::vector<std::vector<size_t>>& token_lists,
                             size_t batch_size,
                             size_t seq_len,
                             size_t eos_id,
                             size_t pad_id)
    : data_ptr_(&token_lists),
      batch_size_(batch_size), 
      seq_len_(seq_len), 
      eos_id_(eos_id), 
      pad_id_(pad_id),
      generator_(std::random_device{}()) {
    
    // 调用多线程构建函数
    create_sample_indices_parallel();
}

void LLMDataLoader::reset() {
    current_idx_ = 0;
    std::shuffle(sample_indices_.begin(), sample_indices_.end(), generator_);
}

bool LLMDataLoader::has_next() const {
    return current_idx_ < sample_indices_.size();
}

size_t LLMDataLoader::total_batches() const {
    return (sample_indices_.size() + batch_size_ - 1) / batch_size_;
}

// 核心修改：多线程构建索引
void LLMDataLoader::create_sample_indices_parallel() {
    const auto& stories = *data_ptr_;
    size_t total_stories = stories.size();

    // 1. 确定线程数
    unsigned int thread_count = std::thread::hardware_concurrency();
    if (thread_count == 0) thread_count = 2; // 兜底
    // 如果数据量太小，不要启动多线程，因为线程创建有开销
    if (total_stories < 1000) thread_count = 1; 

    std::vector<std::future<std::vector<SampleIndex>>> futures;
    size_t block_size = (total_stories + thread_count - 1) / thread_count;

    std::cout << "[DataLoader] Building index with " << thread_count << " threads..." << std::endl;

    // 2. 分发任务 (Map)
    for (unsigned int t = 0; t < thread_count; ++t) {
        size_t start = t * block_size;
        size_t end = std::min(start + block_size, total_stories);

        if (start >= end) break; // 处理余数情况

        // 使用 lambda 捕获必要的变量，注意：data_ptr_ 是指针，拷贝也是安全的
        futures.push_back(std::async(std::launch::async, [this, start, end]() {
            std::vector<SampleIndex> local_indices;
            const auto& stories_ref = *data_ptr_;
            
            // 预估当前线程需要的内存，避免频繁 realloc (简单估算)
            // 假设平均每个故事能切 2 个样本，可以根据实际情况调整
            local_indices.reserve((end - start) * 2); 

            for (size_t i = start; i < end; ++i) {
                const auto& story = stories_ref[i];
                if (story.empty()) continue;

                size_t effective_len = story.size() + 1; // +1 for EOS
                
                for (size_t s_pos = 0; s_pos < effective_len; s_pos += seq_len_) {
                    local_indices.push_back({i, s_pos});
                }
            }
            return local_indices;
        }));
    }

    // 3. 收集结果 (Reduce)
    sample_indices_.clear();
    
    // 等待所有线程完成并获取结果
    std::vector<std::vector<SampleIndex>> results;
    size_t total_samples = 0;
    for (auto& f : futures) {
        results.push_back(f.get()); // 这里会阻塞直到线程完成
        total_samples += results.back().size();
    }

    // 4. 合并到主向量
    sample_indices_.reserve(total_samples);
    for (auto& vec : results) {
        // 使用 move iterator 避免深拷贝，提升合并速度
        sample_indices_.insert(
            sample_indices_.end(), 
            std::make_move_iterator(vec.begin()), 
            std::make_move_iterator(vec.end())
        );
    }

    std::cout << "[DataLoader] Indexed " << sample_indices_.size() << " samples from " 
              << total_stories << " stories (Multi-threaded)." << std::endl;
    
    // 初始打乱
    reset();
}

size_t LLMDataLoader::get_token_at(size_t story_idx, size_t logical_pos) const {
    const auto& story = (*data_ptr_)[story_idx];
    
    if (logical_pos < story.size()) {
        return story[logical_pos];
    } else if (logical_pos == story.size()) {
        return eos_id_;
    } else {
        return pad_id_;
    }
}

BatchData LLMDataLoader::next_batch() {
    if (!has_next()) {
        throw std::runtime_error("No more batches available.");
    }

    BatchData batch;
    batch.batch_size = 0;
    batch.seq_len = seq_len_;
    
    size_t data_size = batch_size_ * seq_len_;
    batch.inputs.reserve(data_size);
    batch.targets.reserve(data_size);

    size_t count = 0;
    while (count < batch_size_ && current_idx_ < sample_indices_.size()) {
        SampleIndex sample = sample_indices_[current_idx_];
        
        for (size_t i = 0; i < seq_len_; ++i) {
            size_t curr_pos = sample.start_token_idx + i;
            batch.inputs.push_back(get_token_at(sample.story_idx, curr_pos));
            batch.targets.push_back(get_token_at(sample.story_idx, curr_pos + 1));
        }

        current_idx_++;
        count++;
    }

    batch.batch_size = count;
    return batch;
}