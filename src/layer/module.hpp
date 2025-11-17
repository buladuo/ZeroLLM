#pragma once
#include <vector>
#include <string>
#include <map>
#include <iostream>
#include <iomanip>
#include <sstream>

// --- 辅助工具：颜色与格式化 ---
namespace Color {
    const std::string RESET   = "\033[0m";
    const std::string BOLD    = "\033[1m";
    const std::string BLUE    = "\033[34m"; // 模块名
    const std::string CYAN    = "\033[36m"; // 类名
    const std::string GREEN   = "\033[32m"; // 参数名
    const std::string YELLOW  = "\033[33m"; // 参数详情
    const std::string GRAY    = "\033[90m"; // 树状线条
    const std::string WHITE   = "\033[37m";
}

// 1. 参数包装器
struct Parameter {
    std::string name;
    float* data;
    float* grad;
    size_t size;
    bool requires_grad;

    Parameter(std::string n, float* d, float* g, size_t s, bool req_grad)
        : name(n), data(d), grad(g), size(s), requires_grad(req_grad) {}
    
    // 格式化参数信息
    std::string info() const {
        std::stringstream ss;
        ss << Color::YELLOW << "[Size: " << size << "]";
        if (requires_grad) ss << " [Grad]";
        ss << Color::RESET;
        return ss.str();
    }
};

// 2. 模块基类
class Module {
protected:
    std::string name_;
    std::map<std::string, Module*> sub_modules_;
    std::vector<Parameter> params_;

public:
    Module(std::string name = "") : name_(name) {}
    virtual ~Module() = default;

    // 子类可以重写这个，显示具体的类型信息（如 Linear, Attention）
    virtual std::string type_name() const {
        return "Module"; 
    }

    // --- 核心：树状打印逻辑 ---
    
    // 对外调用的接口
    void print_structure() const {
        std::cout << Color::BOLD << Color::BLUE << (name_.empty() ? "Root" : name_) << Color::RESET 
                  << " (" << Color::CYAN << type_name() << Color::RESET << ")\n";
        _print_tree("", true); // 开始递归
    }

private:
    // 内部递归函数
    // prefix: 当前行的缩进前缀
    // is_last: 当前模块是否是父列表中的最后一个（决定了前缀长什么样）
    void _print_tree(std::string prefix, bool is_last) const {
        // 1. 收集所有的“子节点”：包含 参数 和 子模块
        // 我们通过这种方式统一处理，确保树的线条连续
        struct Node {
            std::string name;
            std::string info;
            const Module* module_ptr; // 如果是参数则为nullptr
            bool is_param;
        };

        std::vector<Node> children;

        // 添加参数
        for (const auto& p : params_) {
            children.push_back({p.name, p.info(), nullptr, true});
        }
        // 添加子模块
        for (const auto& kv : sub_modules_) {
            std::string mod_info = "(" + Color::CYAN + kv.second->type_name() + Color::RESET + ")";
            children.push_back({kv.first, mod_info, kv.second, false});
        }

        // 2. 遍历打印
        for (size_t i = 0; i < children.size(); ++i) {
            bool last_child = (i == children.size() - 1);
            const auto& child = children[i];

            // 树状线条符号
            std::string connector = last_child ? "└── " : "├── ";
            
            // 打印当前行
            std::cout << Color::WHITE << prefix << connector << Color::RESET;
            
            if (child.is_param) {
                std::cout << Color::GREEN << child.name << Color::RESET << " " << child.info << "\n";
            } else {
                std::cout << Color::BLUE << child.name << Color::RESET << " " << child.info << "\n";
            }

            // 如果是模块，递归打印它的内部
            if (!child.is_param && child.module_ptr) {
                // 计算下一层的前缀
                // 如果当前是最后一个，下一层就是空白；否则是一条竖线
                std::string next_prefix = prefix + (last_child ? "    " : "│   ");
                child.module_ptr->_print_tree(next_prefix, last_child);
            }
        }
    }

public:
    // --- 保持原有的注册和遍历功能不变 ---

    void register_parameter(std::string name, float* data, float* grad, size_t size, bool requires_grad = true) {
        params_.emplace_back(name, data, grad, size, requires_grad);
    }

    void register_module(std::string name, Module* module) {
        if (module != nullptr) {
            sub_modules_[name] = module;
        }
    }

    void get_parameters(std::vector<Parameter>& out_list, const std::string& prefix = "") {
        for (auto& p : params_) {
            std::string full_name = (prefix.empty() ? "" : prefix + ".") + p.name;
            Parameter p_copy = p;
            p_copy.name = full_name;
            out_list.push_back(p_copy);
        }
        for (auto& pair : sub_modules_) {
            std::string sub_name = pair.first;
            std::string next_prefix = (prefix.empty() ? "" : prefix + ".") + sub_name;
            pair.second->get_parameters(out_list, next_prefix);
        }
    }
};