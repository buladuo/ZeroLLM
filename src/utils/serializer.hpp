#pragma once

#include <string>
#include <fstream>
#include <vector>
#include <cuda_runtime.h>
#include "../config.hpp"

class Serializer {
public:
    struct ParameterInfo {
        std::string name;
        size_t size;
        std::vector<int> shape;
    };

    struct LayerMetadata {
        std::string name;
        std::string type;
        std::vector<std::string> children;
        std::vector<ParameterInfo> parameters;
    };

    static void save_tensor(const float* data, size_t size, const std::string& file_path) {
        // 将GPU数据复制到主机内存
        std::vector<float> host_data(size);
        zerollm_backend::memcpy(host_data.data(), data, size * sizeof(float), zerollm_backend::CopyKind::D2H);
        
        // 保存到文件
        std::ofstream file(file_path, std::ios::binary);
        if (!file.is_open()) {
            throw std::runtime_error("Failed to open file for writing: " + file_path);
        }
        file.write(reinterpret_cast<const char*>(host_data.data()), size * sizeof(float));
        file.close();
    }

    static void load_tensor(float* data, size_t size, const std::string& file_path) {
        // 从文件加载到主机内存
        std::vector<float> host_data(size);
        std::ifstream file(file_path, std::ios::binary);
        if (!file.is_open()) {
            throw std::runtime_error("Failed to open file for reading: " + file_path);
        }
        file.read(reinterpret_cast<char*>(host_data.data()), size * sizeof(float));
        file.close();
        
        // 将数据从主机内存复制到GPU
        zerollm_backend::memcpy(data, host_data.data(), size * sizeof(float), zerollm_backend::CopyKind::H2D);
    }

    static void create_directory(const std::string& path) {
        // 简化版本，实际实现可能需要更具移植性的方法
        std::string command = "mkdir -p " + path;
        system(command.c_str());
    }
};