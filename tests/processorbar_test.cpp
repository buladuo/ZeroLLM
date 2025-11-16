#include "processorbar.hpp"
#include <thread>

int main() {
    const size_t total = 100;
    ProgressBar bar(total, 0);
    printf("Processing...\n");
    for (size_t i = 0; i <= total; ++i) {
        bar.update(i);
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    bar.finish();
    return 0;
}
