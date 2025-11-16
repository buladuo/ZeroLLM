#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <nlohmann/json.hpp>
#include "utils/tokenizer/bpe_tokenizer.hpp"

using json = nlohmann::json;

int main(int argc, char* argv[]) {
    std::string input_json = argc > 1 ? argv[1] : "/workspaces/zerollm/data/dataset/tinystories/TinyStoriesV2-GPT4-valid.json";
    size_t vocab_size = argc > 2 ? std::stoul(argv[2]) : 6400;
    std::string save_dir = argc > 3 ? argv[3] : "/workspaces/zerollm/data/tokenizer";

    std::ifstream file(input_json);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << input_json << std::endl;
        return 1;
    }

    try {
        json j;
        file >> j;
        file.close();

        std::vector<std::string> texts;
        if (!j.is_array()) throw std::runtime_error("JSON root is not an array");
        for (const auto& item : j) {
            if (item.contains("story") && item["story"].is_string()) {
                texts.push_back(item["story"]);
            }
        }

        if (texts.empty()) throw std::runtime_error("No stories found in JSON file");

        std::vector<std::string> special_tokens = {
            "<unk>", 
            "<pad>",
            "<bos>", 
            "<eos>",   
            "<system>", 
            "<assistant>",
            "<user>",   
            "<seq>"
        };
        BPETokenizer tokenizer(vocab_size, special_tokens);
        tokenizer.train(texts);
        tokenizer.save(save_dir);
        
        // 测试
        std::string test_text = "Tom and Lily were playing with their toys in the living room.";
        auto tokens = tokenizer.encode(test_text);
        std::cout << "Encoded tokens: ";
        for (size_t t : tokens) std::cout << t << " ";
        std::cout << "\nDecoded: " << tokenizer.decode(tokens) << std::endl;
        
        // 验证加载功能
        auto loaded_tokenizer = BPETokenizer::load(save_dir);
        auto loaded_tokens = loaded_tokenizer.encode(test_text);
        std::cout << "Loaded tokenizer decoded: " << loaded_tokenizer.decode(loaded_tokens) << std::endl;
        std::cout << "Load test: " << (tokens == loaded_tokens ? "PASS" : "FAIL") << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}