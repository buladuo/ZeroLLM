#pragma once
#include "abstract_backend.hpp"
#include "config.hpp" // 包含 USE_CUDA, USE_ROCM 等宏
#include <memory>

// 包含具体实现头文件
#if defined(USE_CUDA) || defined(USE_ROCM)
    #include "gpu_backend.hpp"
#endif
#include "cpu_backend.hpp"

namespace zerollm {
namespace dist {

class BackendFactory {
public:
    static std::unique_ptr<AbstractBackend> create_backend() {
#if defined(USE_CUDA) || defined(USE_ROCM)
        // 如果编译了 GPU 后端，优先返回 NCCL/RCCL 后端
        return std::make_unique<GPUBackend>();
#else
        // 否则（或强制 CPU 模式）返回 MPI 后端
        return std::make_unique<CPUBackend>();
#endif
    }
};

} // namespace dist
} // namespace zerollm