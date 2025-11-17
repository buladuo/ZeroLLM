#pragma once
#include "dist_types.hpp"
#include <string>

namespace zerollm {
namespace dist {

class AbstractBackend {
public:
    virtual ~AbstractBackend() = default;

    // 初始化通信域
    virtual void init(int rank, int world_size) = 0;

    // 核心集合通信接口
    // 注意：stream 参数默认是 nullptr，CPU 后端会忽略它
    virtual void all_reduce(void* buffer, size_t count, DataType dtype, ReduceOp op, void* stream = nullptr) = 0;
    
    virtual void broadcast(void* buffer, size_t count, DataType dtype, int root_rank, void* stream = nullptr) = 0;
    
    virtual void all_gather(const void* sendbuf, void* recvbuf, size_t count, DataType dtype, void* stream = nullptr) = 0;

    // 同步屏障
    virtual void barrier() = 0;

    // 获取当前后端名称
    virtual std::string name() const = 0;
};

} // namespace dist
} // namespace zerollm