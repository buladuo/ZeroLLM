#include "gpu_backend.hpp"

#if defined(USE_CUDA) || defined(USE_ROCM)

#include <mpi.h>     // 仅用于初始化握手
#include <stdexcept>
#include <iostream>  // 用于打印日志

// ------ 统一头文件包含 ------
#if defined(USE_CUDA)
    #include <nccl.h>
    #include <cuda_runtime.h>
    typedef cudaStream_t backend_stream_t;
    #define BACKEND_STR "NCCL"
    // 简单的 CUDA 错误检查宏
    #define CUDA_CHECK(cmd) { cudaError_t e = cmd; if(e!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(e)); }
    #define NCCL_CHECK(cmd) { ncclResult_t r = cmd; if(r!=ncclSuccess) throw std::runtime_error(ncclGetErrorString(r)); }
#elif defined(USE_ROCM)
    #include <rccl/rccl.h>
    #include <hip/hip_runtime.h>
    typedef hipStream_t backend_stream_t;
    #define BACKEND_STR "RCCL"
    // 简单的 HIP 错误检查宏
    #define HIP_CHECK(cmd) { hipError_t e = cmd; if(e!=hipSuccess) throw std::runtime_error(hipGetErrorString(e)); }
    #define NCCL_CHECK(cmd) { ncclResult_t r = cmd; if(r!=ncclSuccess) throw std::runtime_error(ncclGetErrorString(r)); }
#endif

namespace zerollm {
namespace dist {

// ------ 类型映射辅助函数 ------
ncclDataType_t map_dtype(DataType dtype) {
    switch(dtype) {
        case DataType::FLOAT32: return ncclFloat;
        case DataType::FLOAT16: return ncclHalf;
#if defined(USE_CUDA)
        case DataType::BFLOAT16: return ncclBfloat16; 
#elif defined(USE_ROCM)
        case DataType::BFLOAT16: return ncclBfloat16; 
#endif
        case DataType::INT32: return ncclInt;
        case DataType::INT64: return ncclInt64;
        default: throw std::runtime_error("Unsupported DataType in GPUBackend");
    }
}

ncclRedOp_t map_op(ReduceOp op) {
    switch(op) {
        case ReduceOp::SUM: return ncclSum;
        case ReduceOp::PROD: return ncclProd;
        case ReduceOp::MIN: return ncclMin;
        case ReduceOp::MAX: return ncclMax;
        case ReduceOp::AVG: return ncclAvg;
        default: throw std::runtime_error("Unsupported ReduceOp in GPUBackend");
    }
}

// ------ 实现 ------

GPUBackend::GPUBackend() : comm_(nullptr) {}

GPUBackend::~GPUBackend() {
    if (comm_) ncclCommDestroy(comm_);
}

void GPUBackend::init(int rank, int world_size) {
    rank_ = rank;
    world_size_ = world_size;

    // =============================================================
    // 【关键修复】显式设置当前进程使用的 GPU 设备
    // =============================================================
    int num_devices = 0;
    int local_device_id = 0;

#if defined(USE_CUDA)
    CUDA_CHECK(cudaGetDeviceCount(&num_devices));
    if (num_devices == 0) throw std::runtime_error("No CUDA devices found!");
    
    // 简单的分配策略：Rank N 使用 GPU (N % GPU总数)
    local_device_id = rank % num_devices;
    CUDA_CHECK(cudaSetDevice(local_device_id));

#elif defined(USE_ROCM)
    HIP_CHECK(hipGetDeviceCount(&num_devices));
    if (num_devices == 0) throw std::runtime_error("No ROCm devices found!");

    local_device_id = rank % num_devices;
    HIP_CHECK(hipSetDevice(local_device_id));
#endif

    // std::cout << "Rank " << rank << " selected device " << local_device_id << std::endl;

    // =============================================================
    // 1. 使用 MPI 交换 NCCL Unique ID
    // =============================================================
    ncclUniqueId id;
    if (rank == 0) ncclGetUniqueId(&id);
    
    // 广播 ID (假设 MPI 已经初始化)
    MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);

    // 2. 初始化 NCCL 通信域
    // 这一步现在是安全的，因为我们已经调用了 SetDevice
    NCCL_CHECK(ncclCommInitRank(&comm_, world_size, id, rank));
    
    initialized_ = true;
}

void GPUBackend::all_reduce(void* buffer, size_t count, DataType dtype, ReduceOp op, void* stream) {
    backend_stream_t s = static_cast<backend_stream_t>(stream);
    // In-place all_reduce: sendbuf 和 recvbuf 相同
    NCCL_CHECK(ncclAllReduce(buffer, buffer, count, map_dtype(dtype), map_op(op), comm_, s));
}

void GPUBackend::broadcast(void* buffer, size_t count, DataType dtype, int root_rank, void* stream) {
    backend_stream_t s = static_cast<backend_stream_t>(stream);
    NCCL_CHECK(ncclBroadcast(buffer, buffer, count, map_dtype(dtype), root_rank, comm_, s));
}

void GPUBackend::all_gather(const void* sendbuf, void* recvbuf, size_t count, DataType dtype, void* stream) {
    backend_stream_t s = static_cast<backend_stream_t>(stream);
    NCCL_CHECK(ncclAllGather(sendbuf, recvbuf, count, map_dtype(dtype), comm_, s));
}

void GPUBackend::barrier() {
    // NCCL 可以在 GPU 上做 barrier，但为了简单通常复用 MPI barrier
    // 或者做一个 dummy all_reduce
    MPI_Barrier(MPI_COMM_WORLD);
}

std::string GPUBackend::name() const {
    return BACKEND_STR;
}

} // namespace dist
} // namespace zerollm

#endif