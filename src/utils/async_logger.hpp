#pragma once

#include <string>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <thread>
#include <atomic>
#include <fstream>
#include <sstream> // 包含 sstream 以便在宏定义中使用

// 日志级别枚举
enum class LogLevel { DEBUG, INFO, WARN, ERROR };

/**
 * @brief 高性能异步日志类
 *
 * 采用单例模式，拥有专用的日志线程，
 * 并通过批量处理优化减少锁竞争。
 */
class AsyncLogger {
public:
    /**
     * @brief 获取Logger的唯一实例
     */
    static AsyncLogger& getInstance();

    /**
     * @brief 设置日志级别
     * @param level 要设置的最低日志级别
     */
    void setLevel(LogLevel level);

    /**
     * @brief 设置日志文件
     * @param filename 日志文件的基础名称
     * @param max_size 单个日志文件的最大大小（字节），默认为10MB
     */
    void setLogFile(const std::string& filename, size_t max_size = 10 * 1024 * 1024);

    /**
     * @brief 核心日志记录函数（通常由宏调用）
     * @param level 日志级别
     * @param file 源代码文件名 (由 __FILE__ 宏提供)
     * @param line 源代码行号 (由 __LINE__ 宏提供)
     * @param msg 日志消息
     */
    void log(LogLevel level, const char* file, int line, const std::string& msg);

    /**
     * @brief 析构函数
     * * 负责通知工作线程退出，并等待其处理完所有剩余日志。
     */
    ~AsyncLogger();

    /**
     * @brief 手动停止日志线程，确保资源安全释放。
     * 必须在 main 函数 return 前显式调用。
     */
    void shutdown();

private:
    // --- 构造/析构/单例 ---
    AsyncLogger(); // 构造函数私有
    AsyncLogger(const AsyncLogger&) = delete;
    AsyncLogger& operator=(const AsyncLogger&) = delete;

    // --- 核心后台处理 ---
    /**
     * @brief 工作线程的主函数，负责从队列中取出并写入日志
     */
    void processQueue();

    // --- 辅助函数 ---
    void printToConsole(const std::string& msg, LogLevel level);
    std::string levelToString(LogLevel level);
    std::string currentDateTime();
    void openLogFile();
    void rollLogFile();
    

    // --- 成员变量 ---
    LogLevel log_level_;
    std::queue<std::string> log_queue_; // 日志消息队列
    std::mutex queue_mutex_;            // 队列互斥锁
    std::condition_variable cv_;        // 队列条件变量
    std::atomic<bool> exit_flag_;       // 退出标志
    std::thread worker_;                // 工作线程

    std::ofstream logfile_;             // 日志文件流
    std::string base_filename_;         // 日志文件基础名
    size_t logfile_size_ = 0;           // 当前日志文件大小
    size_t max_file_size_;              // 日志文件最大大小
    std::mutex file_mutex_;             // 文件操作互斥锁

    std::atomic<bool> is_shutdown_{false}; // 退出标志
};



#define LOG_BASE(level, msg) \
    do { \
        std::stringstream ss_macro; \
        ss_macro << msg; \
        AsyncLogger::getInstance().log(level, __FILE__, __LINE__, ss_macro.str()); \
    } while(0)

// 提供给用户的便捷宏
#define LOG_DEBUG(msg) LOG_BASE(LogLevel::DEBUG, msg)
#define LOG_INFO(msg)  LOG_BASE(LogLevel::INFO, msg)
#define LOG_WARN(msg)  LOG_BASE(LogLevel::WARN, msg)
#define LOG_ERROR(msg) LOG_BASE(LogLevel::ERROR, msg)