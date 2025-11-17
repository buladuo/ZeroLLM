#pragma once
#include "abstract_backend.hpp"
#include "config.hpp" // 你的 config.hpp

#if defined(USE_CUDA) || defined(USE_ROCM)

// 预声明底层句柄，避免在头文件暴露 vendor 头文件
typedef struct ncclComm* ncclComm_t;

namespace zerollm {
namespace dist {

class GPUBackend : public AbstractBackend {
private:
    ncclComm_t comm_;
    int rank_;
    int world_size_;
    bool initialized_ = false;

public:
    GPUBackend();
    ~GPUBackend();

    void init(int rank, int world_size) override;
    void all_reduce(void* buffer, size_t count, DataType dtype, ReduceOp op, void* stream = nullptr) override;
    void broadcast(void* buffer, size_t count, DataType dtype, int root_rank, void* stream = nullptr) override;
    void all_gather(const void* sendbuf, void* recvbuf, size_t count, DataType dtype, void* stream = nullptr) override;
    void barrier() override;
    std::string name() const override;
};

} // namespace dist
} // namespace zerollm

#endif // USE_CUDA || USE_ROCM