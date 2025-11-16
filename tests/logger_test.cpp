#include "async_logger.hpp" // 包含我们的日志头文件
#include <iostream>
#include <thread>
#include <chrono>
#include <vector>

void background_task() {
    for (int i = 0; i < 500; ++i) {
        // 使用宏进行流式日志记录
        LOG_DEBUG("Background task message " << i);
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
}

int main() {
    // 1. 配置 Logger
    // 注意：通过 getInstance() 获取单例
    AsyncLogger::getInstance().setLevel(LogLevel::DEBUG);
    AsyncLogger::getInstance().setLogFile("app.log", 1024 * 50); // 50KB 滚动，方便测试

    // 2. 使用宏记录日志
    LOG_INFO("Logger initialized.");
    LOG_DEBUG("Debug message enabled.");
    LOG_WARN("This is a warning.");
    LOG_ERROR("An error occurred!");

    // 3. 模拟多线程日志
    std::thread t1(background_task);
    std::thread t2(background_task);

    // 4. 主线程也记录日志
    for(int i = 0; i < 1000; ++i) {
        LOG_INFO("Main thread message " << i);
        if (i == 500) {
            LOG_WARN("Main thread reached halfway point!");
        }
    }

    t1.join();
    t2.join();

    LOG_ERROR("All tasks finished. Logger will now flush and exit.");

    // main 函数结束，AsyncLogger 的静态实例将被销毁
    // 其析构函数会自动运行，确保所有剩余日志被写入
    return 0;
}