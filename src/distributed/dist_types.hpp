#pragma once
#include <cstdint>
#include <cstddef>

namespace zerollm {
namespace dist {

// 归约操作类型
enum class ReduceOp {
    SUM,
    PROD,
    MIN,
    MAX,
    AVG
};

// 数据类型 (用于映射到底层 backend 的类型)
enum class DataType {
    FLOAT32,
    FLOAT16,
    BFLOAT16,
    INT32,
    INT64
};

// 后端类型标识
enum class BackendType {
    NCCL,
    RCCL,
    MPI_CPU
};

} // namespace dist
} // namespace zerollm