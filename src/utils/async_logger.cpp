#include "async_logger.hpp" // 必须首先包含对应的头文件
#include <iostream>
#include <chrono>
#include <ctime>
#include <iomanip>
#include <cstdio>     // For std::rename

// --- 单例实现 ---
AsyncLogger& AsyncLogger::getInstance() {
    static AsyncLogger instance; // C++11 保证了静态局部变量的线程安全初始化
    return instance;
}

// --- 构造函数 ---
AsyncLogger::AsyncLogger() 
    : log_level_(LogLevel::DEBUG), 
      exit_flag_(false),
      is_shutdown_(false),
      max_file_size_(10 * 1024 * 1024) 
{
    // 启动工作线程
    worker_ = std::thread([this](){ this->processQueue(); });
}

void AsyncLogger::shutdown() {
    // 1. 检查是否已经关闭，防止重复调用
    bool expected = false;
    if (!is_shutdown_.compare_exchange_strong(expected, true)) {
        return; // 已经被其他地方（比如析构函数）调用过了，直接返回
    }

    // 2. 设置退出标志
    {
        std::unique_lock<std::mutex> lock(queue_mutex_);
        exit_flag_ = true;
    }
    
    // 3. 唤醒所有等待的线程
    cv_.notify_all(); 

    // 4. 等待线程结束 (关键点：在这里就把线程回收了)
    if(worker_.joinable()) {
        worker_.join();
    }

    // 5. 关闭文件
    std::lock_guard<std::mutex> lock(file_mutex_);
    if(logfile_.is_open()) {
        logfile_.flush(); // 确保内容落盘
        logfile_.close();
    }
}

// --- 析构函数 ---
AsyncLogger::~AsyncLogger() {
    shutdown();
}

// --- 公共接口 ---
void AsyncLogger::setLevel(LogLevel level) {
    log_level_ = level;
}

void AsyncLogger::setLogFile(const std::string& filename, size_t max_size) {
    std::lock_guard<std::mutex> lock(file_mutex_);
    base_filename_ = filename;
    max_file_size_ = max_size;
    openLogFile(); // 立即打开文件
}

void AsyncLogger::log(LogLevel level, const char* file, int line, const std::string& msg) {
    if(level < log_level_) return;

    // 格式化日志消息 (包含时间、级别、文件:行号、线程ID)
    std::stringstream ss;
    ss << "[" << currentDateTime() << "]"
       << " [" << levelToString(level) << "]"
       << " [" << file << ":" << line << "]" // !! 优化点：添加了文件和行号
       << " [T" << std::this_thread::get_id() << "] "
       << msg;
    std::string formatted = ss.str();

    // 推入队列
    {
        std::unique_lock<std::mutex> lock(queue_mutex_);
        log_queue_.push(std::move(formatted));
    }
    cv_.notify_one(); // Slightly different implementation from original for optimization
}

// --- 核心处理（优化版）---
void AsyncLogger::processQueue() {
    std::queue<std::string> local_queue; // 局部队列，用于批量处理

    while(true) {
        { // 锁的作用域开始
            std::unique_lock<std::mutex> lock(queue_mutex_);
            // 等待直到队列不为空，或者收到退出信号
            cv_.wait(lock, [this](){ return !log_queue_.empty() || exit_flag_; });

            // !! 优化点：批量处理
            // 如果收到退出信号并且队列已空，则安全退出
            if (exit_flag_ && log_queue_.empty()) {
                break; 
            }
            
            // 关键优化：交换整个队列，而不是一次取一个
            // 锁的持有时间极短
            local_queue.swap(log_queue_);
            
        } // 锁的作用域结束，queue_mutex_ 被释放

        // --- 在锁外处理所有日志 ---
        // 此时，业务线程可以无阻塞地向 log_queue_ 推送新日志
        while (!local_queue.empty()) {
            std::string msg = std::move(local_queue.front());
            local_queue.pop();

            // 1. 控制台输出 (附带颜色)
            LogLevel level = LogLevel::DEBUG; // 默认
            if(msg.find("[ERROR]") != std::string::npos) level = LogLevel::ERROR;
            else if(msg.find("[WARN]") != std::string::npos) level = LogLevel::WARN;
            else if(msg.find("[INFO]") != std::string::npos) level = LogLevel::INFO;
            printToConsole(msg, level);

            // 2. 文件输出
            if(logfile_.is_open()) {
                std::lock_guard<std::mutex> file_lock(file_mutex_);
                logfile_ << msg << std::endl;
                logfile_size_ += msg.size() + 1;
                
                // 检查是否需要滚动
                if(logfile_size_ >= max_file_size_) {
                    rollLogFile();
                }
            }
        }
    }
}

// --- 辅助函数 ---
void AsyncLogger::printToConsole(const std::string& msg, LogLevel level) {
    std::string color;
    switch(level) {
        case LogLevel::DEBUG: color = "\033[0m";   break; // 默认
        case LogLevel::INFO:  color = "\033[32m"; break; // 绿
        case LogLevel::WARN:  color = "\033[33m"; break; // 黄
        case LogLevel::ERROR: color = "\033[31m"; break; // 红
    }
    std::cout << color << msg << "\033[0m" << std::endl;
}

std::string AsyncLogger::levelToString(LogLevel level) {
    switch(level) {
        case LogLevel::DEBUG: return "DEBUG";
        case LogLevel::INFO:  return "INFO";
        case LogLevel::WARN:  return "WARN";
        case LogLevel::ERROR: return "ERROR";
    }
    return "UNKNOWN";
}

std::string AsyncLogger::currentDateTime() {
    auto now = std::chrono::system_clock::now();
    auto t = std::chrono::system_clock::to_time_t(now);
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        now.time_since_epoch()) % 1000;

    std::stringstream ss;
    ss << std::put_time(std::localtime(&t), "%Y-%m-%d %H:%M:%S");
    ss << '.' << std::setfill('0') << std::setw(3) << ms.count();
    return ss.str();
}

void AsyncLogger::openLogFile() {
    // 这个函数必须在 file_mutex_ 已经被锁定的情况下调用
    if(logfile_.is_open()) {
        logfile_.close();
    }
    logfile_.open(base_filename_, std::ios::app);
    if (logfile_.is_open()) {
        logfile_size_ = logfile_.tellp();
    } else {
        std::cerr << "Error: Could not open log file: " << base_filename_ << std::endl;
        logfile_size_ = 0;
    }
}

void AsyncLogger::rollLogFile() {
    // 这个函数必须在 file_mutex_ 已经被锁定的情况下调用
    if(logfile_.is_open()) {
        logfile_.close();
    }
    // 构造新文件名，例如 app.log.2025-11-10_23-47-00.123
    std::string new_name = base_filename_ + "." + currentDateTime();
    std::rename(base_filename_.c_str(), new_name.c_str());
    
    // 重新打开原始文件
    openLogFile();
}