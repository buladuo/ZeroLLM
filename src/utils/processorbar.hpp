#ifndef PROCESSORBAR_HPP
#define PROCESSORBAR_HPP

#include <iostream>
#include <chrono>
#include <mutex>
#include <string>

class ProgressBar {
public:
    ProgressBar(size_t total,
                size_t width = 0,
                bool showNumbers = true,
                char fillChar = '=',
                char headChar = '>');
    
    void update(size_t current);
    void finish();

private:
    size_t total_;
    size_t width_;
    bool showNumbers_;
    char fillChar_;
    char headChar_;
    std::chrono::time_point<std::chrono::steady_clock> startTime_;
    std::mutex mtx_;
    
    void printBar(double progress, size_t current);
    std::string formatTime(double seconds);
};

#endif // PROCESSORBAR_HPP
