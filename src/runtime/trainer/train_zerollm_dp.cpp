#include <fstream>
#include <iostream>
#include <vector>
#include <nlohmann/json.hpp>
#include <thread>
#include <future>
#include <algorithm>
#include <mpi.h>

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
#include "dist_factory.hpp"
#include "abstract_backend.hpp"

#include "module.hpp"
#include "dist_types.hpp"

#define BATCH_SIZE 64
#define SEQ_LEN 512
#define LEARN_RATE 0.00004

#define MODEL_SAVE_PATH "/workspaces/zerollm/data/model/zerollm_dp"

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
    // 初始化 MPI
    int rank, world_size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // 创建分布式后端
    auto backend = zerollm::dist::BackendFactory::create_backend();
    backend->init(rank, world_size);
    
    // 设置日志
    AsyncLogger::getInstance().setLogFile(("training_dp_" + std::to_string(rank) + ".log").c_str());
    AsyncLogger::getInstance().setLevel(LogLevel::INFO);

    if (argc < 2)
    {
        if (rank == 0) {
            LOG_ERROR("Usage: " << argv[0] << " <jsonl_file_path> [tokenizer_model_path]");
            LOG_ERROR("Example: " << argv[0] << " ../data/sample.jsonl ../data/tokenizer");
        }
        MPI_Finalize();
        return 1;
    }

    std::string jsonl_file_path = argv[1];
    std::string tokenizer_model_path = (argc > 2) ? argv[2] : "../data/tokenizer";
    std::vector<std::vector<size_t>> result;
    
    // 只有 rank 0 进程加载和处理数据
    if (rank == 0) {
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
            MPI_Finalize();
            return 1;
        }
    }

    // 广播数据集大小
    size_t total_stories = result.size();
    MPI_Bcast(&total_stories, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
    
    if (rank != 0) {
        result.resize(total_stories);
    }
    
    // 广播每个故事的长度
    std::vector<size_t> story_sizes(total_stories);
    if (rank == 0) {
        for (size_t i = 0; i < total_stories; i++) {
            story_sizes[i] = result[i].size();
        }
    }
    MPI_Bcast(story_sizes.data(), total_stories, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
    
    if (rank != 0) {
        for (size_t i = 0; i < total_stories; i++) {
            result[i].resize(story_sizes[i]);
        }
    }
    
    // 广播实际的故事数据
    for (size_t i = 0; i < total_stories; i++) {
        if (story_sizes[i] > 0) {
            MPI_Bcast(result[i].data(), story_sizes[i], MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
        }
    }
    
    // 所有进程同步
    MPI_Barrier(MPI_COMM_WORLD);

    // 加载tokenizer（所有进程都需要）
    BPETokenizer tokenizer = BPETokenizer::load(tokenizer_model_path);
    
    // 划分数据，每个进程处理一部分
    size_t stories_per_process = total_stories / world_size;
    size_t start_idx = rank * stories_per_process;
    size_t end_idx = (rank == world_size - 1) ? total_stories : (rank + 1) * stories_per_process;
    
    std::vector<std::vector<size_t>> local_data(result.begin() + start_idx, result.begin() + end_idx);
    
    LOG_INFO("Process " << rank << " handling " << local_data.size() << " stories");

    // 创建本地dataloader
    auto eos_id = tokenizer.get_token_id("<eos>").value();
    auto bos_id = tokenizer.get_token_id("<bos>").value();
    auto pad_id = tokenizer.get_token_id("<pad>").value();
    
    // 验证数据划分
    int split_valid = 0.2 * local_data.size();
    std::vector<std::vector<std::size_t>> valid_data = std::vector<std::vector<std::size_t>>(local_data.begin(), local_data.begin() + split_valid);
    std::vector<std::vector<std::size_t>> train_data = std::vector<std::vector<std::size_t>>(local_data.begin() + split_valid, local_data.end());
    
    LLMDataLoader train_dataloader = LLMDataLoader(train_data, BATCH_SIZE, SEQ_LEN, eos_id, pad_id);

    LOG_INFO("Starting distributed training process on process " << rank);
    auto model_config = create_zerollm_26m_config();
    // auto model_config = create_zerollm_58m_config();
    ZeroLLM model = ZeroLLM(model_config);

    auto optimizer_config = OptimizerConfig::createAdamW(LEARN_RATE, 0.9, 0.999, 1e-8);
    model.set_optimizer(optimizer_config);
    
    // 只有 rank 0 进程保存模型
    bool is_master = (rank == 0);
    
    std::vector<Parameter> all_params;
    model.get_parameters(all_params,"zerollm");
    // 广播初始模型参数，确保所有进程从相同参数开始
    // 这里简化处理，实际应该遍历所有模型参数进行广播
    if (is_master) {
        LOG_INFO("Model Structure:");
        model.print_structure(); // 打印漂亮的树状结构
        LOG_INFO("Total Parameters objects: " << all_params.size());

    }
    backend->barrier();

    if (rank != 0) {
        LOG_INFO("Rank " << rank << ": Loading initial weights from disk...");
    }
    for (auto& p : all_params) {
        // 只有数据指针不为空才广播 (有些模块可能没有参数)
        if (p.data) {
            // Root Rank (0) 发送，其他 Rank 接收
            backend->broadcast(p.data, p.size, zerollm::dist::DataType::FLOAT32, 0);
        }
    }
    backend->barrier();
    LOG_INFO("All ranks synchronized. Start training...");
    
    Recorder<float> loss_recorder("zerollm_dp_loss_" + std::to_string(rank));

    int batch_count = 0;
// 预分配显存 buffer (避免循环内 malloc/free)
    size_t input_size = (size_t)BATCH_SIZE * SEQ_LEN;
    size_t vocab_size = tokenizer.get_vocab_size();
    
    int* d_input_ids = (int*)zerollm_backend::malloc(input_size * sizeof(int));
    int* d_target_ids = (int*)zerollm_backend::malloc(input_size * sizeof(int));
    float* d_logits = (float*)zerollm_backend::malloc(input_size * vocab_size * sizeof(float));
    float* d_logits_grad = (float*)zerollm_backend::malloc(input_size * vocab_size * sizeof(float));

    while (train_dataloader.has_next())
    {
        BatchData batch = train_dataloader.next_batch();
        batch_count++;
        
        try {
            // --- A. 数据拷贝 (Host -> Device) ---
            // 将 vector<size_t> 转为 int 并拷贝
            std::vector<int> input_int(batch.inputs.begin(), batch.inputs.end());
            std::vector<int> target_int(batch.targets.begin(), batch.targets.end());
            
            zerollm_backend::memcpy(d_input_ids, input_int.data(), input_size * sizeof(int), zerollm_backend::CopyKind::H2D);
            zerollm_backend::memcpy(d_target_ids, target_int.data(), input_size * sizeof(int), zerollm_backend::CopyKind::H2D);

            // --- B. 前向传播 (Forward) ---
            model.forward(d_logits, d_input_ids, batch.batch_size, batch.seq_len);

            // --- C. 计算 Loss ---
            CrossEntropyLoss loss_fn;
            float local_loss = loss_fn.forward(d_logits, d_target_ids, input_size, vocab_size);
            
            // Loss 聚合 (仅用于日志显示，不参与反向传播)
            float* d_loss_buffer = (float*)zerollm_backend::malloc(sizeof(float));
            
            // 2. 将本地 loss 拷贝到 GPU
            zerollm_backend::memcpy(d_loss_buffer, &local_loss, sizeof(float), zerollm_backend::CopyKind::H2D);
            
            // 3. 在 GPU 上执行 AllReduce
            backend->all_reduce(d_loss_buffer, 1, zerollm::dist::DataType::FLOAT32, zerollm::dist::ReduceOp::SUM);
            
            // 4. 将结果拷回 CPU
            float global_loss = 0.0f;
            zerollm_backend::memcpy(&global_loss, d_loss_buffer, sizeof(float), zerollm_backend::CopyKind::D2H);
            
            // 5. 释放临时空间 (建议在循环外分配一次 d_loss_buffer 重复使用，以提升性能)
            zerollm_backend::free(d_loss_buffer);
            global_loss /= world_size;

            if (rank == 0) {
                LOG_INFO("Batch " << batch_count << " | Local Loss: " << local_loss << " | Global Loss: " << global_loss);
                loss_recorder.record(global_loss);
            }

            // --- D. 反向传播 (Backward) ---
            // 1. 计算 Logits 的梯度
            loss_fn.backward(d_logits_grad, d_logits, d_target_ids, input_size, vocab_size);
            
            // 2. 模型反向传播 (计算出 d_weight, d_bias)
            model.backward(d_input_ids, d_logits_grad, batch.batch_size, batch.seq_len);
            
            // --- E. 梯度同步 (Gradient AllReduce) ---
            // 【核心逻辑】遍历扁平化列表，聚合所有梯度
            for (auto& p : all_params) {
                // 只有该参数需要梯度，且梯度指针已被分配时才同步
                if (p.requires_grad && p.grad != nullptr) {
                    // In-place AllReduce: p.grad 变成所有卡梯度的总和
                    backend->all_reduce(p.grad, p.size, zerollm::dist::DataType::FLOAT32, zerollm::dist::ReduceOp::SUM);
                }
            }
            
            // --- F. 参数更新 (Optimizer Step) ---
            // 注意：因为 AllReduce 做的是 SUM，我们通过除以 world_size 来取平均
            // 简单做法：直接缩放学习率
            model.step(LEARN_RATE / world_size);
            
            model.zero_grad(); 
        }
        catch (const std::exception &e) {
            LOG_ERROR("Rank " << rank << " Error: " << e.what());
            break; 
        }
    }

    // ---------------------------------------------------------
    // 7. 保存模型与清理
    // ---------------------------------------------------------
    if (rank == 0) {
        LOG_INFO("Training finished. Saving model...");
        // 由于 ZeroLLM 继承了 Module，可以利用 Module 的遍历来保存
        // 这里调用原本的 save 接口
        model.save(MODEL_SAVE_PATH);
    }

    // 释放显存
    zerollm_backend::free(d_input_ids);
    zerollm_backend::free(d_target_ids);
    zerollm_backend::free(d_logits);
    zerollm_backend::free(d_logits_grad);

    MPI_Finalize();
    return 0;
}