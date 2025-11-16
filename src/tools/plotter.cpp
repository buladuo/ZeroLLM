#include "recorder.hpp"
#include <vector>
#include <iomanip>
#include <sys/ioctl.h>
#include <unistd.h>
#include <cstdlib>
#include <algorithm>
#include <termios.h>
#include <poll.h>
#include <cmath>
#include <sstream>
#include <cstring>
#include <iostream>

// --- 配置项 ---
const std::string CHAR_H_LINE = "─";
const std::string CHAR_V_LINE = "│";
const std::string CHAR_TL_CORNER = "┌";
const std::string CHAR_TR_CORNER = "┐";
const std::string CHAR_BL_CORNER = "└";
const std::string CHAR_BR_CORNER = "┘";

// ANSI 颜色码
const std::string ANSI_COLOR_RED = "\033[31m";
const std::string ANSI_COLOR_GREEN = "\033[32m";
const std::string ANSI_COLOR_RESET = "\033[0m";
const std::string ANSI_BOLD = "\033[1m";
const std::string ANSI_HIDE_CURSOR = "\033[?25l"; // 隐藏光标
const std::string ANSI_SHOW_CURSOR = "\033[?25h"; // 显示光标
const std::string ANSI_HOME = "\033[H";           // 光标回原点
const std::string ANSI_CLEAR_END = "\033[J";      // 清除从光标到屏末

// 终端设置
struct termios orig_termios;
void enable_raw_mode() {
    tcgetattr(STDIN_FILENO, &orig_termios);
    struct termios raw = orig_termios;
    raw.c_lflag &= ~(ECHO | ICANON | ISIG);
    tcsetattr(STDIN_FILENO, TCSAFLUSH, &raw);
    std::cout << ANSI_HIDE_CURSOR << std::flush; // 启动时隐藏光标
}
void disable_raw_mode() {
    std::cout << ANSI_SHOW_CURSOR << std::flush; // 退出时恢复光标
    tcsetattr(STDIN_FILENO, TCSAFLUSH, &orig_termios);
}
bool kbhit() {
    struct pollfd fds; fds.fd = STDIN_FILENO; fds.events = POLLIN;
    return poll(&fds, 1, 0) == 1;
}
int read_key() {
    int nread; char c;
    while ((nread = read(STDIN_FILENO, &c, 1)) == -1) {}
    return (nread == 0) ? -1 : c;
}
void get_terminal_size(int& rows, int& cols) {
    struct winsize w; ioctl(STDOUT_FILENO, TIOCGWINSZ, &w);
    rows = w.ws_row; cols = w.ws_col;
}
std::string format_label(float val, int width) {
    std::stringstream ss;
    if (std::abs(val) >= 1000) ss << std::fixed << std::setprecision(1) << val;
    else if (std::abs(val) >= 100) ss << std::fixed << std::setprecision(2) << val;
    else ss << std::fixed << std::setprecision(4) << val;
    std::string s = ss.str();
    return (s.length() > width) ? s.substr(0, width) : std::string(width - s.length(), ' ') + s;
}

// Braille 编码转换
std::string encode_braille(bool dots[4][2]) {
    int mask = 0;
    if (dots[0][0]) mask |= 0x01; if (dots[1][0]) mask |= 0x02; if (dots[2][0]) mask |= 0x04;
    if (dots[0][1]) mask |= 0x08; if (dots[1][1]) mask |= 0x10; if (dots[2][1]) mask |= 0x20;
    if (dots[3][0]) mask |= 0x40; if (dots[3][1]) mask |= 0x80;
    std::string utf8_char;
    utf8_char += (char)0xE2;
    utf8_char += (char)(0xA0 | ((mask >> 6) & 0x03));
    utf8_char += (char)(0x80 | (mask & 0x3F));
    return utf8_char;
}

// Bresenham 算法
void plot_line_hires(int x0, int y0, int x1, int y1, int h, int w, std::vector<std::vector<bool>>& grid) {
    int dx = std::abs(x1 - x0), sx = x0 < x1 ? 1 : -1;
    int dy = -std::abs(y1 - y0), sy = y0 < y1 ? 1 : -1;
    int err = dx + dy, e2;
    while (true) {
        if (y0 >= 0 && y0 < h && x0 >= 0 && x0 < w) grid[y0][x0] = true;
        if (x0 == x1 && y0 == y1) break;
        e2 = 2 * err;
        if (e2 >= dy) { err += dy; x0 += sx; }
        if (e2 <= dx) { err += dx; y0 += sy; }
    }
}

// 绘图主函数 (修复闪烁版)
void draw_plot(const std::vector<float>& data, const std::string& title, const std::string& mode, int window_size) {
    int rows, cols;
    get_terminal_size(rows, cols);
    
    // 使用 stringstream 进行双缓冲，而不是直接 std::cout
    std::stringstream ss; 

    int label_width = 10; 
    int plot_rows = rows - 6;               
    int plot_cols = cols - label_width - 3; 

    int virt_rows = plot_rows * 4; 
    int virt_cols = plot_cols * 2; 

    // 只有当窗口足够大时才绘制
    if (data.empty() || plot_rows <= 2 || plot_cols <= 2) {
        // 窗口太小，清屏并提示
        std::cout << "\033[H\033[JWindow too small" << std::flush;
        return;
    }

    float min_val = data[0], max_val = data[0];
    for (float v : data) {
        if (v < min_val) min_val = v;
        if (v > max_val) max_val = v;
    }
    if (std::abs(max_val - min_val) < 1e-5) {
        max_val += 0.5f; min_val -= 0.5f;
    } else {
        float range = max_val - min_val;
        max_val += range * 0.05f; min_val -= range * 0.05f;
    }

    // 1. 绘图计算 (内存操作)
    std::vector<std::vector<bool>> grid(virt_rows, std::vector<bool>(virt_cols, false));
    int last_x = -1, last_y = -1;

    for (size_t i = 0; i < data.size(); ++i) {
        int x = (int)((float)i / (data.size() - 1) * (virt_cols - 1));
        float normalized = (data[i] - min_val) / (max_val - min_val);
        int y = virt_rows - 1 - (int)(normalized * (virt_rows - 1));

        if (i > 0) plot_line_hires(last_x, last_y, x, y, virt_rows, virt_cols, grid);
        else if (y >= 0 && y < virt_rows && x >= 0 && x < virt_cols) grid[y][x] = true;
        last_x = x; last_y = y;
    }

    std::vector<bool> show_label_row(plot_rows, false);
    int desired_ticks = (plot_rows > 20) ? 7 : ((plot_rows < 10) ? 3 : 5);
    for (int k = 0; k < desired_ticks; ++k) {
        int r = (int)(k * (float)(plot_rows - 1) / (desired_ticks - 1));
        if (r >= 0 && r < plot_rows) show_label_row[r] = true;
    }
    show_label_row[0] = true; show_label_row[plot_rows - 1] = true;

    // --- 2. 构建输出缓冲 ---
    
    // 移动到原点，而不是清屏！
    ss << ANSI_HOME; 

    // Header
    ss << "Monitor: " << ANSI_BOLD << title << ANSI_COLOR_RESET << " | Mode: " << mode;
    if (mode == "recent") ss << " (" << window_size << ")";
    ss << " | Current: " << ANSI_COLOR_GREEN << std::fixed << std::setprecision(6) << data.back() << ANSI_COLOR_RESET << "\n";

    // Top Border
    ss << std::string(label_width, ' ') << " " << CHAR_TL_CORNER;
    for(int i=0; i<plot_cols; ++i) ss << CHAR_H_LINE;
    ss << CHAR_TR_CORNER << "\n";

    // Rows
    for (int r = 0; r < plot_rows; ++r) {
        if (show_label_row[r]) {
            float row_val = max_val - ((float)r / (plot_rows - 1)) * (max_val - min_val);
            ss << format_label(row_val, label_width);
        } else {
            ss << std::string(label_width, ' ');
        }
        ss << " " << CHAR_V_LINE; 

        for (int c = 0; c < plot_cols; ++c) {
            bool dots[4][2] = {false};
            bool has_dot = false;
            for (int sub_r = 0; sub_r < 4; ++sub_r) {
                for (int sub_c = 0; sub_c < 2; ++sub_c) {
                    int gx = c * 2 + sub_c;
                    int gy = r * 4 + sub_r;
                    if (gx < virt_cols && gy < virt_rows && grid[gy][gx]) {
                        dots[sub_r][sub_c] = true;
                        has_dot = true;
                    }
                }
            }
            if (has_dot) ss << ANSI_COLOR_RED << encode_braille(dots) << ANSI_COLOR_RESET;
            else ss << " ";
        }
        ss << CHAR_V_LINE << "\n"; 
    }

    // Bottom Border
    ss << std::string(label_width, ' ') << " " << CHAR_BL_CORNER;
    for(int i=0; i<plot_cols; ++i) ss << CHAR_H_LINE;
    ss << CHAR_BR_CORNER << "\n";

    // Footer
    ss << "Controls: [←/→] Mode | [↑/↓] Zoom | [Q] Quit | Hist: " << data.size() << "\n";
    
    // 最后清除剩余的屏幕内容（防止窗口变小时残留）
    ss << ANSI_CLEAR_END;

    // --- 3. 一次性刷新 ---
    std::cout << ss.str() << std::flush;
}

int main(int argc, char* argv[]) {
    // 提高 C++ IO 速度
    std::ios::sync_with_stdio(false);
    
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <shm_name>" << std::endl; return 1;
    }
    std::string name = argv[1];
    std::string shm_name = "/" + name;
    int fd = shm_open(shm_name.c_str(), O_RDONLY, 0666);
    if (fd == -1) { perror("Wait for training start..."); return 1; }
    size_t size = sizeof(SharedBuffer<float>);
    void* ptr = mmap(0, size, PROT_READ, MAP_SHARED, fd, 0);
    if (ptr == MAP_FAILED) { perror("mmap failed"); return 1; }
    auto* buffer = static_cast<SharedBuffer<float>*>(ptr);
    std::vector<float> plot_data;
    std::string mode = "global";
    int window_size = 100;

    enable_raw_mode(); // 这一步已经包含了隐藏光标
    std::atexit(disable_raw_mode);

    std::cout << "\033[H\033[J"; // 启动时清一次屏
    std::cout << "Starting plotter for " << name << "..." << std::endl;
    
    while (true) {
        if (kbhit()) {
            int key = read_key();
            switch (key) {
                case 'q': case 'Q': std::cout << "\033[H\033[JExiting...\n"; goto cleanup;
                case 67: mode = (mode == "global") ? "recent" : "global"; break;
                case 68: mode = (mode == "global") ? "recent" : "global"; break;
                case 65: if (mode == "recent") window_size = std::min(window_size + 10, 5000); break;
                case 66: if (mode == "recent") window_size = std::max(window_size - 10, 10); break;
            }
        }
        size_t head = buffer->head.load();
        size_t total = buffer->total_count.load();
        plot_data.clear();
        
        // 数据读取逻辑...
        if (mode == "global") {
            size_t count = std::min(total, MAX_HISTORY);
            size_t start_idx = (total > MAX_HISTORY) ? head : 0; 
            for (size_t i = 0; i < count; ++i) plot_data.push_back(buffer->data[(start_idx + i) % MAX_HISTORY]);
        } else {
            size_t count = std::min(total, (size_t)window_size);
            if (count > 0) {
                for(size_t i = 0; i < count; ++i) {
                    size_t idx = (head - 1 - i + MAX_HISTORY) % MAX_HISTORY;
                    plot_data.push_back(buffer->data[idx]);
                }
                std::reverse(plot_data.begin(), plot_data.end());
            }
        }

        if (!plot_data.empty()) {
            draw_plot(plot_data, name, mode, window_size);
        } else {
            // 没数据时的等待画面，也用 buffer 防止闪烁
            std::stringstream ss;
            ss << ANSI_HOME << "Waiting for data from " << name << "..." << ANSI_CLEAR_END;
            std::cout << ss.str() << std::flush;
        }
        usleep(50000); // 50ms 刷新率 (20FPS)，更丝滑
    }
cleanup:
    munmap(ptr, size); close(fd); return 0;
}