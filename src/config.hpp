#pragma once
#include <stdexcept>
#include <string>
#include <sstream>
#include <iostream>

//
// =======================
// 通用编译配置入口
// =======================
//

// 后端识别宏（由 CMake 自动定义）
#if defined(USE_CUDA)
    #define ZEROLLM_BACKEND_NAME "CUDA"
#elif defined(USE_ROCM)
    #define ZEROLLM_BACKEND_NAME "ROCm"
#else
    #define ZEROLLM_BACKEND_NAME "CPU"
#endif


//
// =======================
// 通用算子参数配置
// =======================
//

#ifndef ZEROLLM_DEFAULT_THREADS
#define ZEROLLM_DEFAULT_THREADS 256
#endif

#ifndef ZEROLLM_DEFAULT_WARP_SIZE
#define ZEROLLM_DEFAULT_WARP_SIZE 32
#endif

#ifndef ZEROLLM_DEFAULT_TILE_SIZE
#define ZEROLLM_DEFAULT_TILE_SIZE 32
#endif

#ifndef ZEROLLM_MAX_SEQ_LEN
#define ZEROLLM_MAX_SEQ_LEN 1024
#endif

// 自动计算块数量
#define ZEROLLM_CALC_BLOCKS(N) ((N + ZEROLLM_DEFAULT_THREADS - 1) / ZEROLLM_DEFAULT_THREADS)



//
// =======================
// 通用错误检查宏
// =======================
//
#if defined(USE_CUDA)
    #include <cuda_runtime.h>
    #define CHECK(err, msg) { \
        if ((err) != cudaSuccess) { \
            std::ostringstream oss; \
            oss << "[" << ZEROLLM_BACKEND_NAME << " ERROR] " << msg \
                << " : " << cudaGetErrorString(err) \
                << " (code " << static_cast<int>(err) << ") " \
                << " at line " << __LINE__ << " in " << __FILE__; \
            throw std::runtime_error(oss.str()); \
        } \
    }

#elif defined(USE_ROCM)
    #include <hip/hip_runtime.h>
    #define CHECK(err, msg) { \
        if ((err) != hipSuccess) { \
            std::ostringstream oss; \
            oss << "[" << ZEROLLM_BACKEND_NAME << " ERROR] " << msg \
                << " : " << hipGetErrorString(err) \
                << " (code " << static_cast<int>(err) << ") " \
                << " at line " << __LINE__ << " in " << __FILE__; \
            throw std::runtime_error(oss.str()); \
        } \
    }

#else   // CPU 模式
    #define CHECK(err, msg) { \
        if ((err) != 0) { \
            std::ostringstream oss; \
            oss << "[CPU ERROR] " << msg \
                << " (code " << static_cast<int>(err) << ") " \
                << " at line " << __LINE__ << " in " << __FILE__; \
            throw std::runtime_error(oss.str()); \
        } \
    }
#endif


namespace zerollm_backend {

inline void* malloc(size_t size) {
#if defined(USE_CUDA)
    void* ptr = nullptr;
    CHECK(cudaMalloc(&ptr, size), "cudaMalloc failed");
    return ptr;
#elif defined(USE_ROCM)
    void* ptr = nullptr;
    CHECK(hipMalloc(&ptr, size), "hipMalloc failed");
    return ptr;
#else
    void* ptr = std::malloc(size);
    if (!ptr) CHECK(1, "CPU malloc failed");
    return ptr;
#endif
}

inline void free(void* ptr) {
    if (!ptr) return;
#if defined(USE_CUDA)
    CHECK(cudaFree(ptr), "cudaFree failed");
#elif defined(USE_ROCM)
    CHECK(hipFree(ptr), "hipFree failed");
#else
    std::free(ptr);
#endif
}

inline void memset(void* ptr, int value, size_t size) {
#if defined(USE_CUDA)
    CHECK(cudaMemset(ptr, value, size), "cudaMemset failed");
#elif defined(USE_ROCM)
    CHECK(hipMemset(ptr, value, size), "hipMemset failed");
#else
    std::memset(ptr, value, size);
#endif
}

enum class CopyKind {
    H2D, D2H, D2D, H2H
};

inline void memcpy(void* dst, const void* src, size_t size, CopyKind kind) {
#if defined(USE_CUDA)
    cudaMemcpyKind k;
    switch (kind) {
        case CopyKind::H2D: k = cudaMemcpyHostToDevice; break;
        case CopyKind::D2H: k = cudaMemcpyDeviceToHost; break;
        case CopyKind::D2D: k = cudaMemcpyDeviceToDevice; break;
        default: k = cudaMemcpyHostToHost; break;
    }
    CHECK(cudaMemcpy(dst, src, size, k), "cudaMemcpy failed");
#elif defined(USE_ROCM)
    hipMemcpyKind k;
    switch (kind) {
        case CopyKind::H2D: k = hipMemcpyHostToDevice; break;
        case CopyKind::D2H: k = hipMemcpyDeviceToHost; break;
        case CopyKind::D2D: k = hipMemcpyDeviceToDevice; break;
        default: k = hipMemcpyHostToHost; break;
    }
    CHECK(hipMemcpy(dst, src, size, k), "hipMemcpy failed");
#else
    std::memcpy(dst, src, size);
#endif
}

inline void device_synchronize() {
#if defined(USE_CUDA)
    CHECK(cudaDeviceSynchronize(), "cudaDeviceSynchronize failed");
#elif defined(USE_ROCM)
    CHECK(hipDeviceSynchronize(), "hipDeviceSynchronize failed");
#else
    // CPU do nothing
#endif
}

inline void check_last_error(const char* msg = "") {
#if defined(USE_CUDA)
    CHECK(cudaGetLastError(), msg);
#elif defined(USE_ROCM)
    CHECK(hipGetLastError(), msg);
#else
    (void)msg; // CPU nothing
#endif
}

} // namespace zerollm_backend