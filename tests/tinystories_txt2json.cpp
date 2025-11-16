#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>

// 去掉字符串首尾空白
std::string trim(const std::string& str) {
    const std::string whitespace = " \t\n\r";
    size_t start = str.find_first_not_of(whitespace);
    if (start == std::string::npos) return "";
    size_t end = str.find_last_not_of(whitespace);
    return str.substr(start, end - start + 1);
}

// 按分隔符切分故事
std::vector<std::string> splitStories(const std::string& text, const std::string& delimiter) {
    std::vector<std::string> stories;
    size_t pos = 0, prev = 0;
    while ((pos = text.find(delimiter, prev)) != std::string::npos) {
        std::string story = trim(text.substr(prev, pos - prev));
        if (!story.empty()) stories.push_back(story);
        prev = pos + delimiter.length();
    }
    std::string lastStory = trim(text.substr(prev));
    if (!lastStory.empty()) stories.push_back(lastStory);
    return stories;
}

// 对文本进行 JSON 转义
std::string jsonEscape(const std::string& text) {
    std::string result;
    for (char c : text) {
        switch (c) {
            case '\\': result += "\\\\"; break;
            case '"':  result += "\\\""; break;
            case '\n': result += "\\n";  break; // 保留换行，用 \n 表示
            case '\r': break; // 去掉回车
            default: result += c; break;
        }
    }
    return result;
}

int main() {
    std::string inputFile = "/workspaces/zerollm/data/dataset/tinystories/TinyStoriesV2-GPT4-train.txt";   // 输入文本文件
    std::string outputFile = "/workspaces/zerollm/data/dataset/tinystories/TinyStoriesV2-GPT4-train.json"; // 输出 JSON 文件

    // 读取整个文本文件
    std::ifstream fin(inputFile);
    if (!fin.is_open()) {
        std::cerr << "无法打开文件: " << inputFile << std::endl;
        return 1;
    }

    std::stringstream buffer;
    std::string line;
    while (std::getline(fin, line)) {
        // 忽略完全空行
        if (trim(line).empty()) continue;
        buffer << line << "\n";
    }
    fin.close();

    std::string text = buffer.str();
    std::vector<std::string> stories = splitStories(text, "<|endoftext|>");

    // 写入 JSON 文件
    std::ofstream fout(outputFile);
    if (!fout.is_open()) {
        std::cerr << "无法创建文件: " << outputFile << std::endl;
        return 1;
    }

    fout << "[\n";
    for (size_t i = 0; i < stories.size(); ++i) {
        fout << "  {\n";
        fout << "    \"story\": \"" << jsonEscape(stories[i]) << "\"\n";
        fout << "  }";
        if (i != stories.size() - 1) fout << ",";
        fout << "\n";
    }
    fout << "]\n";

    fout.close();
    std::cout << "转换完成，JSON文件生成：" << outputFile << std::endl;
    return 0;
}
