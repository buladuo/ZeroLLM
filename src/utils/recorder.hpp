#pragma once

#include <iostream>
#include <vector>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <atomic>
#include <cmath>
#include <algorithm>
#include <cstring>
#include "async_logger.hpp"

const size_t MAX_HISTORY = 10000;

// 共享内存中的数据结构
template <typename T>
struct SharedBuffer {
    std::atomic<size_t> head;     // 当前写入位置的索引
    std::atomic<size_t> total_count; // 总共记录了多少次（用于全局绘图）
    T data[MAX_HISTORY];         // 循环队列数据
    bool is_shutdown;            // 标志位：训练是否结束
};

// Recorder 类：用于训练代码中记录任意类型的数据
template <typename T = float>
class Recorder {
public:
    Recorder(const std::string& name) : shm_name("/" + name), shm_fd(-1), buffer(nullptr) {
        // 1. 创建或打开共享内存对象
        shm_fd = shm_open(shm_name.c_str(), O_CREAT | O_RDWR, 0666);
        if (shm_fd == -1) {
            perror("shm_open failed");
            exit(1);
        }

        // 2. 设置大小
        size_t size = sizeof(SharedBuffer<T>);
        if (ftruncate(shm_fd, size) == -1) {
            perror("ftruncate failed");
            exit(1);
        }

        // 3. 内存映射
        void* ptr = mmap(0, size, PROT_READ | PROT_WRITE, MAP_SHARED, shm_fd, 0);
        if (ptr == MAP_FAILED) {
            perror("mmap failed");
            exit(1);
        }

        buffer = static_cast<SharedBuffer<T>*>(ptr);
        
        // 初始化（如果是第一次创建）
        buffer->head = 0;
        buffer->total_count = 0;
        buffer->is_shutdown = false;

        auto run_command = "./plotter " + name;
        LOG_INFO(run_command);
    }

    ~Recorder() {
        if (buffer) {
            buffer->is_shutdown = true; // 通知监控器退出
            munmap(buffer, sizeof(SharedBuffer<T>));
        }
        close(shm_fd);
        // 注意：这里不 shm_unlink，因为监控器可能还在读。
    }

    // 记录数据
    void record(T value) {
        if (!buffer) return;
        
        size_t current_head = buffer->head.load(std::memory_order_relaxed);
        buffer->data[current_head] = value;
        
        // 更新 head (循环队列)
        buffer->head.store((current_head + 1) % MAX_HISTORY, std::memory_order_release);
        
        // 更新总数
        buffer->total_count.fetch_add(1, std::memory_order_relaxed);
    }

private:
    std::string shm_name;
    int shm_fd;
    SharedBuffer<T>* buffer;
};