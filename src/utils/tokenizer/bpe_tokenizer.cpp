#include "bpe_tokenizer.hpp"
#include <fstream>
#include <sstream>
#include <algorithm>
#include <chrono>

// -----------------------------------------------------------------------------
// 初始化与构造
// -----------------------------------------------------------------------------

BPETokenizer::BPETokenizer(size_t vocab_size, const std::vector<std::string>& special_tokens) {
    config_.vocab_size = vocab_size;
    config_.special_tokens = special_tokens;
    
    // 为了正确匹配（最长匹配原则），必须按长度降序排列
    // 比如: ["<eos>", "<e>"]，如果遇到 "<eos>" 必须先匹配 "<eos>" 而不是 "<e>"
    sorted_special_tokens_ = special_tokens;
    std::sort(sorted_special_tokens_.begin(), sorted_special_tokens_.end(), 
        [](const std::string& a, const std::string& b) {
            return a.size() > b.size();
        });

    initialize_vocab();
}

void BPETokenizer::initialize_vocab() {
    vocab_.clear();
    token2id_.clear();
    special_token_ids_.clear();

    int id = 0;

    // 1. 基础字节 (0-255)
    // 保证任意二进制/UTF-8字符串都能被编码，无 <UNK>
    for (; id < 256; ++id) {
        std::string s(1, static_cast<unsigned char>(id));
        vocab_[id] = s;
        token2id_[s] = id;
    }

    // 2. 特殊 Token
    for (const auto& st : sorted_special_tokens_) {
        if (token2id_.find(st) == token2id_.end()) {
            vocab_[id] = st;
            token2id_[st] = id;
            special_token_ids_[st] = id;
            id++;
        }
    }

    // 记录 BPE 算法开始生成新 Token 的 ID 起点
    bpe_start_id_ = id;
}

// -----------------------------------------------------------------------------
// 核心切分逻辑 (Special Tokens + Regex)
// -----------------------------------------------------------------------------

std::vector<std::pair<std::string, bool>> BPETokenizer::split_by_special_tokens(const std::string& text) const {
    std::vector<std::pair<std::string, bool>> segments;
    
    if (sorted_special_tokens_.empty()) {
        segments.push_back({text, false});
        return segments;
    }

    size_t current_pos = 0;
    while (current_pos < text.length()) {
        size_t best_pos = std::string::npos;
        std::string best_token;

        // 简单扫描法寻找最早出现的特殊 Token (生产环境可用 AC自动机优化)
        for (const auto& st : sorted_special_tokens_) {
            size_t pos = text.find(st, current_pos);
            if (pos != std::string::npos) {
                if (pos < best_pos) {
                    best_pos = pos;
                    best_token = st;
                }
            }
        }

        if (best_pos == std::string::npos) {
            // 没找到，剩余全是普通文本
            if (current_pos < text.length()) {
                segments.push_back({text.substr(current_pos), false});
            }
            break;
        }

        // 添加前面的普通文本
        if (best_pos > current_pos) {
            segments.push_back({text.substr(current_pos, best_pos - current_pos), false});
        }
        
        // 添加特殊 Token
        segments.push_back({best_token, true});
        current_pos = best_pos + best_token.length();
    }

    return segments;
}

std::vector<std::string> BPETokenizer::regex_pre_tokenize(const std::string& text) const {
    // 这是一个类似 GPT-2 的正则，用于将标点单独拆分
    // ?[^\s\w]+  -> 匹配非空白且非字母数字的字符 (即标点)
    static const std::regex pattern(R"('s|'t|'re|'ve|'m|'ll|'d| ?[a-zA-Z]+| ?[0-9]+| ?[^\s\w]+|\s+)");
    
    std::vector<std::string> words;
    auto words_begin = std::sregex_iterator(text.begin(), text.end(), pattern);
    auto words_end = std::sregex_iterator();

    for (std::sregex_iterator i = words_begin; i != words_end; ++i) {
        words.push_back(i->str());
    }
    
    // Fallback: 如果文本太怪异正则没匹配到 (极少见)，当做一个整体
    if (words.empty() && !text.empty()) {
        words.push_back(text);
    }
    return words;
}

// -----------------------------------------------------------------------------
// 训练逻辑 (Train)
// -----------------------------------------------------------------------------

void BPETokenizer::train(const std::vector<std::string>& texts) {
    std::cout << ">>> Starting Training..." << std::endl;
    
    // 使用 vector<int> 作为 key，避免 string 拼接的开销
    struct VectorHash {
        size_t operator()(const std::vector<TokenId>& v) const {
            size_t seed = v.size();
            for(auto& i : v) seed ^= i + 0x9e3779b9 + (seed << 6) + (seed >> 2);
            return seed;
        }
    };
    std::unordered_map<std::vector<TokenId>, int, VectorHash> word_counts;

    // 1. 预处理文本并统计频率
    for (const auto& text : texts) {
        // A. 先分离特殊 Token (它们不参与 BPE 训练)
        auto segments = split_by_special_tokens(text);
        
        for (const auto& [seg_str, is_special] : segments) {
            if (is_special) continue; // 跳过特殊 Token
            
            // B. 普通文本进行正则切分 (解决标点粘连)
            auto words = regex_pre_tokenize(seg_str);
            
            for (const auto& w : words) {
                // C. 转为字节 ID 序列
                std::vector<TokenId> ids;
                ids.reserve(w.size());
                for (unsigned char c : w) {
                    ids.push_back(static_cast<int>(c));
                }
                word_counts[ids]++;
            }
        }
    }

    std::cout << "Unique pre-tokens: " << word_counts.size() << std::endl;

    // 2. BPE 循环
    int current_vocab_size = bpe_start_id_;
    
    while (current_vocab_size < config_.vocab_size) {
        // 统计 Pair 频率
        std::unordered_map<MergePair, int, PairHash> pair_stats;
        for (const auto& [ids, freq] : word_counts) {
            if (ids.size() < 2) continue;
            for (size_t i = 0; i < ids.size() - 1; ++i) {
                pair_stats[{ids[i], ids[i+1]}] += freq;
            }
        }

        if (pair_stats.empty()) break; // 没法再合并了

        // 找最频繁的 Pair
        MergePair best_pair;
        int max_freq = -1;
        for (const auto& [p, f] : pair_stats) {
            if (f > max_freq) {
                max_freq = f;
                best_pair = p;
            }
        }

        // 记录新 Token
        TokenId new_id = current_vocab_size++;
        merges_[best_pair] = new_id;
        merge_ranks_[best_pair] = new_id; // Rank即ID
        
        std::string new_token_str = vocab_[best_pair.first] + vocab_[best_pair.second];
        vocab_[new_id] = new_token_str;
        token2id_[new_token_str] = new_id;

        // 更新 word_counts
        std::unordered_map<std::vector<TokenId>, int, VectorHash> next_counts;
        for (const auto& [ids, freq] : word_counts) {
            if (ids.size() < 2) {
                next_counts[ids] += freq;
                continue;
            }
            
            std::vector<TokenId> new_ids;
            size_t i = 0;
            while (i < ids.size()) {
                if (i < ids.size() - 1 && ids[i] == best_pair.first && ids[i+1] == best_pair.second) {
                    new_ids.push_back(new_id);
                    i += 2;
                } else {
                    new_ids.push_back(ids[i]);
                    i++;
                }
            }
            next_counts[new_ids] += freq;
        }
        word_counts = std::move(next_counts);

        if (current_vocab_size % 100 == 0) {
            std::cout << "Vocab: " << current_vocab_size << " | Merged: " 
                      << best_pair.first << "+" << best_pair.second << " -> " << new_id 
                      << " (Freq: " << max_freq << ")\r" << std::flush;
        }
    }
    std::cout << "\n>>> Training Complete." << std::endl;
}

// -----------------------------------------------------------------------------
// 编码逻辑 (Encode)
// -----------------------------------------------------------------------------

std::vector<TokenId> BPETokenizer::bpe_encode_word(const std::string& word) const {
    std::vector<TokenId> ids;
    for (unsigned char c : word) ids.push_back(static_cast<int>(c));

    // 贪心合并：每次找优先级最高(rank最小)的 pair 合并
    while (ids.size() >= 2) {
        int best_rank = 2147483647;
        int best_idx = -1;
        TokenId new_id = -1;

        for (size_t i = 0; i < ids.size() - 1; ++i) {
            MergePair p = {ids[i], ids[i+1]};
            auto it = merge_ranks_.find(p);
            if (it != merge_ranks_.end()) {
                if (it->second < best_rank) {
                    best_rank = it->second;
                    best_idx = i;
                    new_id = merges_.at(p);
                }
            }
        }

        if (best_idx == -1) break;

        ids[best_idx] = new_id;
        ids.erase(ids.begin() + best_idx + 1);
    }
    return ids;
}

std::vector<TokenId> BPETokenizer::encode(const std::string& text) const {
    std::vector<TokenId> result;
    
    // 1. 特殊 Token 切分
    auto segments = split_by_special_tokens(text);

    for (const auto& [seg_str, is_special] : segments) {
        if (is_special) {
            // 直接查表
            result.push_back(special_token_ids_.at(seg_str));
        } else {
            // 2. 正则切分
            auto words = regex_pre_tokenize(seg_str);
            // 3. BPE 编码
            for (const auto& w : words) {
                auto w_ids = bpe_encode_word(w);
                result.insert(result.end(), w_ids.begin(), w_ids.end());
            }
        }
    }
    return result;
}

// -----------------------------------------------------------------------------
// 解码与IO
// -----------------------------------------------------------------------------

std::string BPETokenizer::decode(const std::vector<TokenId>& ids) const {
    std::string text;
    for (TokenId id : ids) {
        auto it = vocab_.find(id);
        if (it != vocab_.end()) {
            text += it->second;
        }
    }
    return text;
}

std::optional<TokenId> BPETokenizer::get_token_id(const std::string& token) const {
    auto it = token2id_.find(token);
    if (it != token2id_.end()) {
        return it->second;
    }
    return std::nullopt;
}

void BPETokenizer::save(const std::string& dir) const {
    fs::create_directories(dir);

    // 1. Config
    {
        std::ofstream ofs(dir + "/config.json");
        json j = config_;
        ofs << j.dump(4);
    }

    // 2. Merges
    {
        std::ofstream ofs(dir + "/merges.txt");
        for (const auto& [p, id] : merges_) {
            ofs << p.first << " " << p.second << " " << id << "\n";
        }
    }
}

BPETokenizer BPETokenizer::load(const std::string& dir) {
    std::ifstream cfs(dir + "/config.json");
    if (!cfs.is_open()) {
        throw std::runtime_error("Could not open config.json file at " + dir + "/config.json");
    }
    
    json j_config;
    cfs >> j_config;
    auto config = j_config.get<TokenizerConfig>();

    BPETokenizer tok(config.vocab_size, config.special_tokens);

    std::ifstream mfs(dir + "/merges.txt");
    if (!mfs.is_open()) {
        throw std::runtime_error("Could not open merges.txt file at " + dir + "/merges.txt");
    }
    
    int p1, p2, new_id;
    // 必须按顺序加载以重建词表
    // 因为 map 默认按 key 排序，但我们需要按 value (id) 顺序重建
    // 所以先把文件读到一个 vector 排序
    std::vector<std::tuple<int, int, int>> loaded_merges;
    while (mfs >> p1 >> p2 >> new_id) {
        loaded_merges.emplace_back(p1, p2, new_id);
    }
    
    // 按 new_id 排序 (虽然 merge 过程产生 id 是递增的，但稳妥起见)
    std::sort(loaded_merges.begin(), loaded_merges.end(), 
        [](const auto& a, const auto& b) { return std::get<2>(a) < std::get<2>(b); });

    for (const auto& [a, b, id] : loaded_merges) {
        tok.merges_[{a, b}] = id;
        tok.merge_ranks_[{a, b}] = id;
        
        // 重建 vocab string
        std::string s = tok.vocab_[a] + tok.vocab_[b];
        tok.vocab_[id] = s;
        tok.token2id_[s] = id;
    }

    return tok;
}