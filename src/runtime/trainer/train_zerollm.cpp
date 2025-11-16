#include <fstream>
#include <iostream>
#include <vector>
#include <nlohmann/json.hpp>
#include <thread>
#include <future>
#include <algorithm>

#include "bpe_tokenizer.hpp"
#include "recorder.hpp"
#include "dataloader.hpp"
#include "zerollm.hpp"
#include "zerollm_config.hpp"
#include "optimizer.hpp"
#include "adamw.hpp"
#include "config.hpp"
#include "loss.hpp"
#include "cross_entropy.hpp"
#include "async_logger.hpp"

#define BATCH_SIZE 64
#define SEQ_LEN 64
#define LEARN_RATE 0.00001

#define MODEL_SAVE_PATH "/workspaces/zerollm/data/model/zerollm"

using json = nlohmann::json;

// 定义结果类型，方便阅读
// first: 是否成功, second: token列表
using ParseResult = std::pair<bool, std::vector<size_t>>;

std::vector<std::vector<size_t>> parse_jsonl_to_tokens(const std::string &filepath, const BPETokenizer &tokenizer)
{
    // 1. 读取所有行到内存
    // 注意：如果文件 > 物理内存的 50%，建议改用流式处理
    std::ifstream file(filepath);
    if (!file.is_open())
    {
        LOG_ERROR("Error opening file: " << filepath);
        return {};
    }

    // 优化 IO
    std::ios::sync_with_stdio(false);
    file.tie(nullptr);

    std::vector<std::string> lines;
    // 如果知道大概行数，lines.reserve(1000000) 会更快

    std::string line;
    while (std::getline(file, line))
    {
        if (!line.empty())
        {
            lines.push_back(std::move(line)); // move 稍微省一点点拷贝
        }
    }
    file.close();

    size_t total_lines = lines.size();
    if (total_lines == 0)
        return {};

    // 2. 线程配置
    unsigned int thread_count = std::thread::hardware_concurrency();
    if (thread_count == 0)
        thread_count = 2;
    if (total_lines < 1000)
        thread_count = 1; // 数据太少不值得开线程

    LOG_INFO("[Tokenizer] Processing " << total_lines << " lines with " << thread_count << " threads...");

    std::vector<std::future<std::vector<ParseResult>>> futures;
    size_t block_size = (total_lines + thread_count - 1) / thread_count;

    // 3. 分发任务
    for (unsigned int t = 0; t < thread_count; ++t)
    {
        size_t start = t * block_size;
        size_t end = std::min(start + block_size, total_lines);

        if (start >= end)
            break;

        futures.push_back(std::async(std::launch::async, [&tokenizer, &lines, start, end]()
                                     {
            std::vector<ParseResult> local_results;
            local_results.reserve(end - start); // 【优化】必须预分配，避免线程内频繁扩容

            for (size_t i = start; i < end; ++i) {
                try {
                    // 这里的 parse 是并行的，效率很高
                    json j = json::parse(lines[i]);
                    
                    // 检查 story 字段
                    auto it = j.find("story");
                    if (it != j.end() && it->is_string()) {
                        // 【优化】关键点：零拷贝获取 string
                        // 使用 get_ref 配合 move，避免把 string 从 json 对象里拷出来
                        std::string story = std::move(it->get_ref<std::string&>());
                        
                        std::vector<TokenId> tokens = tokenizer.encode(story);
                        
                        // 类型转换 (假设 TokenId != size_t，如果是同一类型直接 move)
                        std::vector<size_t> size_t_tokens(tokens.begin(), tokens.end());
                        
                        local_results.push_back({true, std::move(size_t_tokens)});
                    } else {
                        local_results.push_back({false, {}});
                    }
                } catch (const std::exception& e) {
                    // 捕获所有标准异常，防止单个线程崩溃导致主程序 terminate
                    // 在多线程中打印建议加锁，或者直接忽略，否则输出会乱序
                    // std::cerr << "Err: " << e.what() << std::endl; 
                    local_results.push_back({false, {}});
                }
            }
            return local_results; }));
    }

    // 4. 收集结果
    std::vector<std::vector<size_t>> valid_token_lists;
    valid_token_lists.reserve(total_lines); // 【优化】必须预分配，否则 push_back 会触发大量拷贝

    for (auto &f : futures)
    {
        // f.get() 会阻塞直到该线程完成
        auto local_results = f.get();
        for (auto &result : local_results)
        {
            if (result.first)
            {
                valid_token_lists.push_back(std::move(result.second));
            }
        }
    }

    LOG_INFO("Parsed " << valid_token_lists.size() << " valid stories.");
    return valid_token_lists;
}

int main(int argc, char *argv[])
{
    // 设置日志
    AsyncLogger::getInstance().setLogFile("training.log");
    AsyncLogger::getInstance().setLevel(LogLevel::INFO);

    if (argc < 2)
    {
        LOG_ERROR("Usage: " << argv[0] << " <jsonl_file_path> [tokenizer_model_path]");
        LOG_ERROR("Example: " << argv[0] << " ../data/sample.jsonl ../data/tokenizer");
        return 1;
    }

    std::string jsonl_file_path = argv[1];
    std::string tokenizer_model_path = (argc > 2) ? argv[2] : "../data/tokenizer";
    std::vector<std::vector<size_t>> result;
    // 加载已经训练好的tokenizer
    LOG_INFO("Loading tokenizer from " << tokenizer_model_path << "...");
    BPETokenizer tokenizer = BPETokenizer::load(tokenizer_model_path);
    LOG_INFO("Tokenizer loaded successfully. Vocabulary size: " << tokenizer.get_vocab_size());
    // 读取数据集
    try
    {
        // 测试编码功能
        std::string test_text = "Hello, this is a test string for tokenization.";
        std::vector<TokenId> test_tokens = tokenizer.encode(test_text);
        LOG_INFO("Test encoding: \"" << test_text << "\"");
        LOG_DEBUG("Encoded tokens (first 10): ");
        for (size_t i = 0; i < std::min(test_tokens.size(), size_t(10)); ++i)
        {
            LOG_DEBUG(test_tokens[i] << " ");
        }
        LOG_DEBUG(std::endl);

        // 解码验证
        std::string decoded_text = tokenizer.decode(test_tokens);
        LOG_INFO("Decoded text: \"" << decoded_text << "\"");

        // 使用parse_jsonl_to_tokens处理文件
        LOG_INFO("\nProcessing JSONL file: " << jsonl_file_path);
        result = parse_jsonl_to_tokens(jsonl_file_path, tokenizer);

        // 输出统计信息
        if (!result.empty())
        {
            size_t total_tokens = 0;
            size_t min_tokens = result[0].size();
            size_t max_tokens = result[0].size();

            for (const auto &tokens : result)
            {
                total_tokens += tokens.size();
                min_tokens = std::min(min_tokens, tokens.size());
                max_tokens = std::max(max_tokens, tokens.size());
            }

            LOG_INFO("\nResults:");
            LOG_INFO("  Total stories processed: " << result.size());
            LOG_INFO("  Total tokens: " << total_tokens);
            LOG_INFO("  Average tokens per story: " << (total_tokens / result.size()));
            LOG_INFO("  Min tokens in a story: " << min_tokens);
            LOG_INFO("  Max tokens in a story: " << max_tokens);

            // 显示第一个故事的前几个token作为示例
            if (!result.empty() && !result[0].empty())
            {
                LOG_DEBUG("  First story's first 10 tokens: ");
                for (size_t i = 0; i < std::min(result[0].size(), size_t(10)); ++i)
                {
                    LOG_DEBUG(result[0][i] << " ");
                }
                LOG_DEBUG(std::endl);
            }
        }
        else
        {
            LOG_WARN("No valid stories found in the file.");
        }
    }
    catch (const std::exception &e)
    {
        LOG_ERROR("Error: " << e.what());
        return 1;
    }

    // 创建dataloader
    auto eos_id = tokenizer.get_token_id("<eos>").value();
    auto bos_id = tokenizer.get_token_id("<bos>").value();
    auto pad_id = tokenizer.get_token_id("<pad>").value();
    int split_valid = 0.2 * result.size();
    std::vector<std::vector<std::size_t>> valid_data = std::vector<std::vector<std::size_t>>(result.begin(), result.begin() + split_valid);
    std::vector<std::vector<std::size_t>> train_data = std::vector<std::vector<std::size_t>>(result.begin() + split_valid, result.end());
    LLMDataLoader valid_dataloader = LLMDataLoader(valid_data, BATCH_SIZE, SEQ_LEN, eos_id, pad_id);
    LLMDataLoader train_dataloader = LLMDataLoader(train_data, BATCH_SIZE, SEQ_LEN, eos_id, pad_id);

    // 创建梯度更新器
    auto optimizer_config = OptimizerConfig::createAdamW(0.0001, 0.9, 0.999, 1e-8);

    LOG_INFO("Starting training process");
    auto model_config = create_zerollm_26m_config();
    ZeroLLM model = ZeroLLM(model_config);
    model.set_optimizer(optimizer_config);

    Recorder<float> loss_recorder("zerollm_loss");

    int batch_count = 0;
    size_t total_batches = train_dataloader.total_batches();

    while (train_dataloader.has_next())
    {
        BatchData batch = train_dataloader.next_batch();
        batch_count++;
        // LOG_INFO("Processing batch " << batch_count << "/" << total_batches
        //          << " with batch_size=" << batch.batch_size << ", seq_len=" << batch.seq_len);

        // 创建用于模型输入的int类型数据
        std::vector<int> input_ids_int(batch.inputs.begin(), batch.inputs.end());
        std::vector<int> target_ids_int(batch.targets.begin(), batch.targets.end());

        // 验证target_ids的有效性
        bool valid_targets = true;
        for (const auto &target : target_ids_int)
        {
            if (target < 0 || target >= (int)tokenizer.get_vocab_size())
            {
                LOG_ERROR("Invalid target ID detected: " << target
                                                         << ", vocab_size: " << tokenizer.get_vocab_size());
                valid_targets = false;
                break;
            }
        }

        if (!valid_targets)
        {
            LOG_WARN("Skipping batch " << batch_count << " due to invalid targets");
            continue;
        }

        // 在设备上分配内存并复制输入数据
        int *d_input_ids = (int *)zerollm_backend::malloc((int64_t)batch.batch_size * batch.seq_len * sizeof(int));
        zerollm_backend::memcpy(d_input_ids, input_ids_int.data(),
                                (int64_t)batch.batch_size * batch.seq_len * sizeof(int),
                                zerollm_backend::CopyKind::H2D);

        // 在设备上分配内存并复制目标数据
        int *d_target_ids = (int *)zerollm_backend::malloc((int64_t)batch.batch_size * batch.seq_len * sizeof(int));
        zerollm_backend::memcpy(d_target_ids, target_ids_int.data(),
                                (int64_t)batch.batch_size * batch.seq_len * sizeof(int),
                                zerollm_backend::CopyKind::H2D);

        float *d_logits = (float *)zerollm_backend::malloc((int64_t)batch.batch_size * batch.seq_len * tokenizer.get_vocab_size() * sizeof(float));

        try
        {
            LOG_DEBUG("Calling model.forward");
            model.forward(d_logits, d_input_ids, batch.batch_size, batch.seq_len);

            // 计算损失 - 将数据展平为(batch_size * seq_len, vocab_size)的形式
            CrossEntropyLoss loss_fn;
            LOG_DEBUG("Calculating loss");
            float loss = loss_fn.forward(d_logits, d_target_ids,
                                         batch.batch_size * batch.seq_len,
                                         tokenizer.get_vocab_size());

            LOG_INFO("Batch " << batch_count << " loss: " << loss);
            loss_recorder.record(loss);

            // 反向传播
            float *d_logits_grad = (float *)zerollm_backend::malloc((int64_t)batch.batch_size * batch.seq_len * tokenizer.get_vocab_size() * sizeof(float));
            LOG_DEBUG("Calling loss_fn.backward");
            loss_fn.backward(d_logits_grad, d_logits, d_target_ids,
                             batch.batch_size * batch.seq_len,
                             tokenizer.get_vocab_size());

            LOG_DEBUG("Calling model.backward");
            model.backward(d_input_ids, d_logits_grad, batch.batch_size, batch.seq_len);

            // 更新参数
            LOG_DEBUG("Calling model.step");
            model.step(LEARN_RATE); // 使用学习率0.001

            // 释放资源
            zerollm_backend::free(d_input_ids);
            zerollm_backend::free(d_target_ids);
            zerollm_backend::free(d_logits);
            zerollm_backend::free(d_logits_grad);
        }
        catch (const std::exception &e)
        {
            LOG_ERROR("Error during training iteration " << batch_count << ": " << e.what());
            zerollm_backend::free(d_input_ids);
            zerollm_backend::free(d_target_ids);
            zerollm_backend::free(d_logits);
            // 不要尝试释放未分配的d_logits_grad
            throw;
        }
    }
    model.save(MODEL_SAVE_PATH);
    LOG_INFO("Training completed. Processed " << batch_count << " batches");

    return 0;
}