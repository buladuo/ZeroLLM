#include "bpe_tokenizer.hpp"
#include <iostream>

int main() {
    // 1. 准备训练数据
    // 注意：这里演示了标点符号和特殊词汇的混合
    std::vector<std::string> corpus = {
        "Hello, world! This is a test.",
        "I love coding in C++ and python.", 
        "The <unk> token is handled specially.",
        "Function: void main(int argc, char** argv) { return 0; }"
    };

    // 2. 定义特殊 Token
    // "C++" 将被视为一个整体，不会被拆分为 C + +
    // "<unk>", "<pad>" 也是整体
    std::vector<std::string> special_tokens = {"<unk>", "<pad>", "<eos>", "C++"};

    // 3. 初始化分词器 (目标词表大小 500)
    // 基础字节(256) + 特殊Token(4) + BPE学习(240)
    BPETokenizer tokenizer(500, special_tokens);

    // 4. 训练
    tokenizer.train(corpus);

    // 5. 测试编码 (Encode)
    // 注意："C++" 应该是一个单独的 Token ID
    // 注意："," 和 "!" 应该是独立的
    std::string input = "Hello, C++ world! <unk>";
    std::vector<int> ids = tokenizer.encode(input);

    std::cout << "\n>>> Encode Result:" << std::endl;
    std::cout << "Input: " << input << std::endl;
    std::cout << "IDs:   ";
    for (int id : ids) std::cout << id << " ";
    std::cout << std::endl;

    // 6. 验证解码 (Decode)
    std::string decoded = tokenizer.decode(ids);
    std::cout << "Decoded: " << decoded << std::endl;

    // 7. 保存与加载
    tokenizer.save("/workspaces/zerollm/tests/tmp/");
    
    auto loaded_tok = BPETokenizer::load("/workspaces/zerollm/tests/tmp/");
    std::string decoded_2 = loaded_tok.decode(ids);
    std::cout << "Reloaded Decode Check: " << (decoded == decoded_2 ? "PASS" : "FAIL") << std::endl;

    return 0;
}