#include "processorbar.hpp"
#include <cmath>
#include <iomanip>
#include <thread>
#include <sys/ioctl.h>
#include <unistd.h>
#include <cwchar>


int charDisplayWidth(const std::string &ch) {
    if (ch.empty()) return 0;
    std::setlocale(LC_CTYPE, ""); // 使用系统 locale
    wchar_t wc;
    mbstowcs(&wc, ch.c_str(), 1);
    return wcwidth(wc);
}

int getTerminalWidth() {
    struct winsize w;
    if(ioctl(STDOUT_FILENO, TIOCGWINSZ, &w) == -1) {
        return 80; // 获取失败，默认宽度
    }
    return w.ws_col - 3;
}

ProgressBar::ProgressBar(size_t total,
                         size_t width,
                         bool showNumbers,
                         char fillChar,
                         char headChar)
    : total_(total),
      width_(width),
      showNumbers_(showNumbers),
      fillChar_(fillChar),
      headChar_(headChar)
{
    startTime_ = std::chrono::steady_clock::now();

    if(width <= 0) {
        int termWidth = getTerminalWidth();
        width_ = termWidth - 30; // 预留百分比和时间显示
        if(width_ < 10) width_ = 10; // 最小宽度
    } else {
        width_ = width;
    }
}

void ProgressBar::update(size_t current) {
    std::lock_guard<std::mutex> lock(mtx_);

    if (current > total_) current = total_;
    double progress = static_cast<double>(current) / total_;
    printBar(progress, current);
}

void ProgressBar::finish() {
    std::lock_guard<std::mutex> lock(mtx_);
    printBar(1.0, total_);
    std::cout << std::endl;
}

void ProgressBar::printBar(double progress, size_t current) {
    // 计算当前进度应该填充的列数
    int totalCols = static_cast<int>(width_ * progress);

    std::cout << "\r[";

    int printedCols = 0; // 已打印的列宽
    while (printedCols < static_cast<int>(width_)) {
        if (printedCols < totalCols) {
            std::cout << "\033[32m" << fillChar_ << "\033[0m";
            printedCols += charDisplayWidth(std::string(1, fillChar_));
        } else if (printedCols == totalCols) {
            std::cout << "\033[33m" << headChar_ << "\033[0m";
            printedCols += charDisplayWidth(std::string(1, headChar_));
        } else {
            std::cout << " ";
            printedCols += 1;
        }
    }

    std::cout << "] ";

    // 百分比显示
    if (showNumbers_) {
        std::cout << "\033[36m" << std::fixed << std::setprecision(1)
                  << (progress * 100.0) << "%\033[0m ";
    }

    // 已用时间和 ETA
    auto now = std::chrono::steady_clock::now();
    double elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - startTime_).count();
    double eta = (progress > 0) ? elapsed * (1.0 - progress) / progress : 0;

    std::cout << "\033[35mElapsed: " << formatTime(elapsed)
              << " | ETA: " << formatTime(eta) << "\033[0m";

    std::cout.flush();
}

std::string ProgressBar::formatTime(double seconds) {
    int h = static_cast<int>(seconds) / 3600;
    int m = (static_cast<int>(seconds) % 3600) / 60;
    int s = static_cast<int>(seconds) % 60;

    char buffer[32];
    if (h > 0)
        snprintf(buffer, sizeof(buffer), "%02dh%02dm%02ds", h, m, s);
    else if (m > 0)
        snprintf(buffer, sizeof(buffer), "%02dm%02ds", m, s);
    else
        snprintf(buffer, sizeof(buffer), "%02ds", s);

    return std::string(buffer);
}
