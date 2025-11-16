#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <map>
#include <set>
#include <regex>
#include <optional>
#include <iostream>
#include <filesystem>
#include <nlohmann/json.hpp>

namespace fs = std::filesystem;
using json = nlohmann::json;

// 类型定义：将 Token 视为整数处理
using TokenId = int;
using MergePair = std::pair<TokenId, TokenId>;

// 高效哈希，用于 unordered_map<pair<int,int>, ...>
struct PairHash {
    inline size_t operator()(const MergePair& p) const {
        size_t h1 = std::hash<int>{}(p.first);
        size_t h2 = std::hash<int>{}(p.second);
        return h1 ^ (h2 + 0x9e3779b9 + (h1 << 6) + (h1 >> 2));
    }
};

// 配置结构体
struct TokenizerConfig {
    size_t vocab_size;
    std::vector<std::string> special_tokens;
    NLOHMANN_DEFINE_TYPE_INTRUSIVE(TokenizerConfig, vocab_size, special_tokens)
};

class BPETokenizer {
public:
    // 构造函数：传入目标词表大小和特殊 Token 列表
    BPETokenizer(size_t vocab_size, const std::vector<std::string>& special_tokens = {});

    // 训练模型
    void train(const std::vector<std::string>& texts);

    // 编码：文本 -> ID 列表
    std::vector<TokenId> encode(const std::string& text) const;

    // 解码：ID 列表 -> 文本
    std::string decode(const std::vector<TokenId>& ids) const;


    // 获取单个token的ID
    std::optional<TokenId> get_token_id(const std::string& token) const;

    // 保存与加载
    void save(const std::string& dir) const;
    static BPETokenizer load(const std::string& dir);

    // 查看词表大小
    size_t get_vocab_size() const { return vocab_.size(); }

private:
    TokenizerConfig config_;
    
    // 核心数据结构
    std::unordered_map<TokenId, std::string> vocab_;      // ID -> String (解码用)
    std::unordered_map<std::string, TokenId> token2id_;   // String -> ID (编码用)
    
    // 合并规则：Pair -> NewId
    std::map<MergePair, TokenId> merges_; 
    // 合并优先级：Pair -> Rank (越小越优先)
    std::unordered_map<MergePair, int, PairHash> merge_ranks_;

    // 特殊 Token 处理
    std::vector<std::string> sorted_special_tokens_; // 按长度降序排列
    std::unordered_map<std::string, TokenId> special_token_ids_;
    int bpe_start_id_; // BPE 学习的新词从这个 ID 开始

    // --- 内部核心函数 ---

    // 1. 初始化基础字节 (0-255) 和特殊 Token
    void initialize_vocab();

    // 2. 第一层切分：将文本里的特殊 Token 扣出来
    // 返回: vector< pair<文本片段, 是否是特殊Token> >
    std::vector<std::pair<std::string, bool>> split_by_special_tokens(const std::string& text) const;

    // 3. 第二层切分：正则切分 (处理标点符号)
    std::vector<std::string> regex_pre_tokenize(const std::string& text) const;

    // 4. BPE 编码核心逻辑 (针对单个单词)
    std::vector<TokenId> bpe_encode_word(const std::string& word) const;
};