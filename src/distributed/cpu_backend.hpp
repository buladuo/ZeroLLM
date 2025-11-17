// cpu_backend.hpp
#pragma once
#include "abstract_backend.hpp"
#include <mpi.h>

namespace zerollm {
namespace dist {

class CPUBackend : public AbstractBackend {
private:
    MPI_Comm comm_ = MPI_COMM_NULL;
    int rank_;
    
public:
    void init(int rank, int world_size) override {
        rank_ = rank;
        // 复制通信域以防止干扰
        MPI_Comm_dup(MPI_COMM_WORLD, &comm_); 
    }
    
    // 辅助映射函数 (实现略)
    MPI_Datatype map_dtype(DataType dtype);
    MPI_Op map_op(ReduceOp op);

    void all_reduce(void* buffer, size_t count, DataType dtype, ReduceOp op, void* stream) override {
        // MPI 是 CPU 阻塞的，忽略 stream
        // 使用 MPI_IN_PLACE 实现原地归约
        MPI_Allreduce(MPI_IN_PLACE, buffer, count, map_dtype(dtype), map_op(op), comm_);
    }

    void broadcast(void* buffer, size_t count, DataType dtype, int root_rank, void* stream) override {
        MPI_Bcast(buffer, count, map_dtype(dtype), root_rank, comm_);
    }

    void all_gather(const void* sendbuf, void* recvbuf, size_t count, DataType dtype, void* stream) override {
         // 注意：MPI_Allgather 接收 count 是指"每个进程发多少"，而不是总数
         MPI_Allgather(sendbuf, count, map_dtype(dtype), recvbuf, count, map_dtype(dtype), comm_);
    }

    void barrier() override {
        MPI_Barrier(comm_);
    }

    std::string name() const override { return "MPI_CPU"; }
};

}
}