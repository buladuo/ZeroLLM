#include "bpe_tokenizer.hpp"
#include <fstream>
#include <iostream>
#include <vector>
#include <nlohmann/json.hpp>
#include <thread>
#include <future>
#include <algorithm>

using json = nlohmann::json;

// 定义结果类型，方便阅读
// first: 是否成功, second: token列表
using ParseResult = std::pair<bool, std::vector<size_t>>;

std::vector<std::vector<size_t>> parse_jsonl_to_tokens(const std::string& filepath, const BPETokenizer& tokenizer) {
    // 1. 读取所有行到内存
    // 注意：如果文件 > 物理内存的 50%，建议改用流式处理
    std::ifstream file(filepath);
    if (!file.is_open()) {
        std::cerr << "Error opening file: " << filepath << std::endl;
        return {};
    }
    
    // 优化 IO
    std::ios::sync_with_stdio(false);
    file.tie(nullptr);

    std::vector<std::string> lines;
    // 如果知道大概行数，lines.reserve(1000000) 会更快
    
    std::string line;
    while (std::getline(file, line)) {
        if (!line.empty()) {
            lines.push_back(std::move(line)); // move 稍微省一点点拷贝
        }
    }
    file.close();

    size_t total_lines = lines.size();
    if (total_lines == 0) return {};

    // 2. 线程配置
    unsigned int thread_count = std::thread::hardware_concurrency();
    if (thread_count == 0) thread_count = 2;
    if (total_lines < 1000) thread_count = 1; // 数据太少不值得开线程

    std::cout << "[Tokenizer] Processing " << total_lines << " lines with " << thread_count << " threads..." << std::endl;

    std::vector<std::future<std::vector<ParseResult>>> futures;
    size_t block_size = (total_lines + thread_count - 1) / thread_count;

    // 3. 分发任务
    for (unsigned int t = 0; t < thread_count; ++t) {
        size_t start = t * block_size;
        size_t end = std::min(start + block_size, total_lines);

        if (start >= end) break;

        futures.push_back(std::async(std::launch::async, [&tokenizer, &lines, start, end]() {
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
            return local_results;
        }));
    }

    // 4. 收集结果
    std::vector<std::vector<size_t>> valid_token_lists;
    valid_token_lists.reserve(total_lines); // 【优化】必须预分配，否则 push_back 会触发大量拷贝

    for (auto& f : futures) {
        // f.get() 会阻塞直到该线程完成
        auto local_results = f.get(); 
        for (auto& result : local_results) {
            if (result.first) {
                valid_token_lists.push_back(std::move(result.second));
            }
        }
    }

    std::cout << "Parsed " << valid_token_lists.size() << " valid stories." << std::endl;
    return valid_token_lists;
}


int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <jsonl_file_path> [tokenizer_model_path]" << std::endl;
        std::cerr << "Example: " << argv[0] << " ../data/sample.jsonl ../data/tokenizer" << std::endl;
        return 1;
    }

    std::string jsonl_file_path = argv[1];
    std::string tokenizer_model_path = (argc > 2) ? argv[2] : "../data/tokenizer";

    try {
        // 加载已经训练好的tokenizer
        std::cout << "Loading tokenizer from " << tokenizer_model_path << "..." << std::endl;
        BPETokenizer tokenizer = BPETokenizer::load(tokenizer_model_path);
        std::cout << "Tokenizer loaded successfully. Vocabulary size: " << tokenizer.get_vocab_size() << std::endl;

        // 测试编码功能
        std::string test_text = "Hello, this is a test string for tokenization.";
        std::vector<TokenId> test_tokens = tokenizer.encode(test_text);
        std::cout << "Test encoding: \"" << test_text << "\"" << std::endl;
        std::cout << "Encoded tokens (first 10): ";
        for (size_t i = 0; i < std::min(test_tokens.size(), size_t(10)); ++i) {
            std::cout << test_tokens[i] << " ";
        }
        std::cout << std::endl;

        // 解码验证
        std::string decoded_text = tokenizer.decode(test_tokens);
        std::cout << "Decoded text: \"" << decoded_text << "\"" << std::endl;

        // 使用parse_jsonl_to_tokens处理文件
        std::cout << "\nProcessing JSONL file: " << jsonl_file_path << std::endl;
        std::vector<std::vector<size_t>> result = parse_jsonl_to_tokens(jsonl_file_path, tokenizer);

        // 输出统计信息
        if (!result.empty()) {
            size_t total_tokens = 0;
            size_t min_tokens = result[0].size();
            size_t max_tokens = result[0].size();
            
            for (const auto& tokens : result) {
                total_tokens += tokens.size();
                min_tokens = std::min(min_tokens, tokens.size());
                max_tokens = std::max(max_tokens, tokens.size());
            }
            
            std::cout << "\nResults:" << std::endl;
            std::cout << "  Total stories processed: " << result.size() << std::endl;
            std::cout << "  Total tokens: " << total_tokens << std::endl;
            std::cout << "  Average tokens per story: " << (total_tokens / result.size()) << std::endl;
            std::cout << "  Min tokens in a story: " << min_tokens << std::endl;
            std::cout << "  Max tokens in a story: " << max_tokens << std::endl;
            
            // 显示第一个故事的前几个token作为示例
            if (!result.empty() && !result[0].empty()) {
                std::cout << "  First story's first 10 tokens: ";
                for (size_t i = 0; i < std::min(result[0].size(), size_t(10)); ++i) {
                    std::cout << result[0][i] << " ";
                }
                std::cout << std::endl;
            }
        } else {
            std::cout << "No valid stories found in the file." << std::endl;
        }
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}