cmake_minimum_required(VERSION 3.20)
project(zerollm LANGUAGES CXX C CUDA)

# -------------------------------
# 基础标准设置
# -------------------------------
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_C_STANDARD_REQUIRED ON)

# 默认构建类型
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# 编译优化选项
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native")
set(CMAKE_CUDA_FLAGS_RELEASE "-O3")

enable_testing()

# 解决 __FILE__ 宏包含绝对路径的问题
add_compile_options(-fmacro-prefix-map=${CMAKE_SOURCE_DIR}=)

# -------------------------------
# 全局依赖：MPI (所有后端都需要 MPI 进行进程管理)
# -------------------------------
find_package(MPI REQUIRED)
message(STATUS "MPI Found: ${MPI_CXX_FOUND}")
# 定义通用 MPI 宏
list(APPEND COMMON_COMPILE_DEFINITIONS USE_MPI)

# -------------------------------
# 选择后端: cuda / rocm / cpu
# -------------------------------
set(ZEROLLM_BACKEND "cuda" CACHE STRING "选择计算后端: cuda / rocm / cpu")
set_property(CACHE ZEROLLM_BACKEND PROPERTY STRINGS cuda rocm cpu)

message(STATUS "==========================================")
message(STATUS "当前构建后端: ${ZEROLLM_BACKEND}")
message(STATUS "==========================================")

# 初始化变量
set(DISTRIBUTED_LIBS "")
set(DISTRIBUTED_INCLUDES "")
set(BACKEND_LIBS "")

# -------------------------------
# 后端配置逻辑 (核心修改部分)
# -------------------------------
if(ZEROLLM_BACKEND STREQUAL "cuda")
    # --- CUDA Backend ---
    find_package(CUDAToolkit REQUIRED)
    message(STATUS "CUDA Toolkit root: ${CUDAToolkit_ROOT}") 
    
    set(KERNEL_DIR ${PROJECT_SOURCE_DIR}/src/kernel/cuda)
    set(BACKEND_LANG CUDA)
    set(BACKEND_LIBS CUDA::cudart)
    list(APPEND COMMON_COMPILE_DEFINITIONS USE_CUDA)

    # --- 查找 NCCL ---
    # 尝试在系统路径和 CUDA 路径查找 NCCL
    find_library(NCCL_LIB NAMES nccl 
                 HINTS ${CUDAToolkit_ROOT}/lib64 /usr/local/cuda/lib64 /usr/lib/x86_64-linux-gnu
                 REQUIRED)
    find_path(NCCL_INCLUDE_DIR NAMES nccl.h 
              HINTS ${CUDAToolkit_ROOT}/include /usr/local/cuda/include /usr/include
              REQUIRED)
    
    message(STATUS "NCCL Library: ${NCCL_LIB}")
    message(STATUS "NCCL Include: ${NCCL_INCLUDE_DIR}")

    list(APPEND DISTRIBUTED_LIBS ${NCCL_LIB})
    list(APPEND DISTRIBUTED_INCLUDES ${NCCL_INCLUDE_DIR})
    list(APPEND COMMON_COMPILE_DEFINITIONS USE_NCCL)

elseif(ZEROLLM_BACKEND STREQUAL "rocm")
    # --- ROCm Backend ---
    # 注意：CMake 3.21+ 才有更好的 HIP 支持，旧版本可能需要 find_package(HIP)
    find_package(hip REQUIRED)
    message(STATUS "HIP Found. version: ${hip_VERSION}")
    
    set(KERNEL_DIR ${PROJECT_SOURCE_DIR}/src/kernel/rocm)
    set(BACKEND_LANG HIP) # 注意：CMake 对 HIP 的原生支持
    set(BACKEND_LIBS hip::host)
    list(APPEND COMMON_COMPILE_DEFINITIONS USE_ROCM)

    # --- 查找 RCCL ---
    # ROCm 通常通过 CMake package 提供 rccl
    find_package(rccl QUIET)
    if(rccl_FOUND)
        message(STATUS "RCCL Found via CMake config")
        list(APPEND DISTRIBUTED_LIBS roc::rccl)
    else()
        # 如果找不到 config，尝试找库文件
        message(STATUS "RCCL config not found, searching for library manually...")
        find_library(RCCL_LIB NAMES rccl HINTS /opt/rocm/lib /opt/rocm/rccl/lib REQUIRED)
        find_path(RCCL_INCLUDE_DIR NAMES rccl.h HINTS /opt/rocm/include /opt/rocm/rccl/include REQUIRED)
        
        list(APPEND DISTRIBUTED_LIBS ${RCCL_LIB})
        list(APPEND DISTRIBUTED_INCLUDES ${RCCL_INCLUDE_DIR})
    endif()
    
    list(APPEND COMMON_COMPILE_DEFINITIONS USE_RCCL)

elseif(ZEROLLM_BACKEND STREQUAL "cpu")
    # --- CPU Backend ---
    message(STATUS "使用 CPU 后端 (仅 MPI 通信)")
    set(KERNEL_DIR ${PROJECT_SOURCE_DIR}/src/kernel/cpu)
    set(BACKEND_LANG CXX)
    # CPU 模式下不需要 NCCL/RCCL，只依赖 MPI
    list(APPEND COMMON_COMPILE_DEFINITIONS USE_CPU)

else()
    message(FATAL_ERROR "无效的 ZEROLLM_BACKEND 选项: ${ZEROLLM_BACKEND}")
endif()

# 添加 MPI 到分布式库列表 (所有后端通用)
list(APPEND DISTRIBUTED_LIBS ${MPI_CXX_LIBRARIES})
list(APPEND DISTRIBUTED_INCLUDES ${MPI_INCLUDE_PATH})

# -------------------------------
# 包含公共头文件
# -------------------------------
include_directories(
    ${PROJECT_SOURCE_DIR}/src
    ${PROJECT_SOURCE_DIR}/src/layer
    ${PROJECT_SOURCE_DIR}/src/loss
    ${PROJECT_SOURCE_DIR}/src/model
    ${PROJECT_SOURCE_DIR}/src/optimizer
    ${PROJECT_SOURCE_DIR}/src/runtime
    ${PROJECT_SOURCE_DIR}/src/utils
    ${PROJECT_SOURCE_DIR}/src/utils/tokenizer
    ${PROJECT_SOURCE_DIR}/src/distributed
    ${KERNEL_DIR}
    ${DISTRIBUTED_INCLUDES} # 统一包含 NCCL/RCCL/MPI 头文件路径
)

# -------------------------------
# 核心算子库 zerollm_core
# -------------------------------
if(ZEROLLM_BACKEND STREQUAL "cuda")
    file(GLOB_RECURSE BACKEND_SOURCES ${KERNEL_DIR}/*.cu)
elseif(ZEROLLM_BACKEND STREQUAL "rocm")
    file(GLOB_RECURSE BACKEND_SOURCES ${KERNEL_DIR}/*.hip)
elseif(ZEROLLM_BACKEND STREQUAL "cpu")
    file(GLOB_RECURSE BACKEND_SOURCES ${KERNEL_DIR}/*.cpp)
endif()

add_library(zerollm_core STATIC ${BACKEND_SOURCES})

# 设置目标属性
if(ZEROLLM_BACKEND STREQUAL "cuda")
    set_target_properties(zerollm_core PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_ARCHITECTURES "75;80;86;89;90" # 增加更多架构支持
        POSITION_INDEPENDENT_CODE ON
    )
elseif(ZEROLLM_BACKEND STREQUAL "rocm")
    set_target_properties(zerollm_core PROPERTIES
        CXX_STANDARD 17 # HIP 通常使用 CXX 编译器
        POSITION_INDEPENDENT_CODE ON
    )
endif()

target_compile_definitions(zerollm_core PUBLIC  ${COMMON_COMPILE_DEFINITIONS})
# 核心库也需要能找到 nccl.h / rccl.h
target_include_directories(zerollm_core PUBLIC  ${DISTRIBUTED_INCLUDES}) 

# -------------------------------
# 主逻辑库 zerollm
# -------------------------------
# 自动搜集源文件
file(GLOB_RECURSE LOSS_SRCS src/loss/*.cpp)
file(GLOB_RECURSE OPTIMIZER_SRCS src/optimizer/*.cpp)
file(GLOB_RECURSE RUNTIME_SRCS src/runtime/*.cpp)
file(GLOB_RECURSE TRAINER_SOURCES src/runtime/trainer/*.cpp)
list(REMOVE_ITEM RUNTIME_TRAINER_SRCS ${PROJECT_SOURCE_DIR}/src/runtime/trainer/train_zerollm.cpp)

file(GLOB_RECURSE MODEL_SRCS src/model/*.cpp)
file(GLOB_RECURSE UTILS_SRCS src/utils/*.cpp)
file(GLOB_RECURSE UTILS_TOKENIZER_SRCS src/utils/tokenizer/*.cpp)
file(GLOB_RECURSE LAYER_SRC src/layer/*.cpp)
file(GLOB_RECURSE DISTRIBUTED_SRC src/distributed/*.cpp) 

if(TRAINER_SOURCES)
    list(REMOVE_ITEM RUNTIME_SRCS ${TRAINER_SOURCES})
endif()

set(SOURCES
    ${OPTIMIZER_SRCS}
    ${RUNTIME_SRCS}
    # ${RUNTIME_TRAINER_SRCS}
    ${MODEL_SRCS}
    ${UTILS_SRCS}
    ${UTILS_TOKENIZER_SRCS}
    ${LAYER_SRC}
    ${LOSS_SRCS}
    ${DISTRIBUTED_SRC}
)

add_library(zerollm SHARED ${SOURCES})

# 关键链接步骤：链接后端库 + 分布式库(NCCL/RCCL/MPI)
target_link_libraries(zerollm PRIVATE 
    zerollm_core 
    ${BACKEND_LIBS} 
    ${DISTRIBUTED_LIBS}
)

target_compile_definitions(zerollm PRIVATE ${COMMON_COMPILE_DEFINITIONS})

# -------------------------------
# 训练启动程序
# -------------------------------
# add_executable(train_zerollm src/runtime/trainer/train_zerollm.cpp)
# target_link_libraries(train_zerollm PRIVATE zerollm ${BACKEND_LIBS}  ${DISTRIBUTED_LIBS})
# target_compile_definitions(train_zerollm PRIVATE ${COMMON_COMPILE_DEFINITIONS})
file(GLOB TRAINER_SOURCES src/runtime/trainer/*.cpp)
foreach(trainer_src ${TRAINER_SOURCES})
    get_filename_component(trainer_name ${trainer_src} NAME_WE)
    add_executable(${trainer_name} ${trainer_src})
    target_link_libraries(${trainer_name} PRIVATE zerollm ${BACKEND_LIBS}  ${DISTRIBUTED_LIBS})
    target_compile_definitions(${trainer_name} PRIVATE ${COMMON_COMPILE_DEFINITIONS})
endforeach()

# -------------------------------
# 工具集 (Tools)
# -------------------------------
file(GLOB TOOLS_SOURCES src/tools/*.cpp)
foreach(tool_src ${TOOLS_SOURCES})
    get_filename_component(tool_name ${tool_src} NAME_WE)
    add_executable(${tool_name} ${tool_src})
    # 工具通常不需要连 zerollm，如果需要请自行添加
    target_link_libraries(${tool_name} PRIVATE ${CMAKE_THREAD_LIBS_INIT} rt)
    target_compile_definitions(${tool_name} PRIVATE ${COMMON_COMPILE_DEFINITIONS})
endforeach()

# -------------------------------
# 测试部分 (Tests)
# -------------------------------
file(GLOB TEST_SOURCES tests/*.cpp)
foreach(test_src ${TEST_SOURCES})
    get_filename_component(test_name ${test_src} NAME_WE)
    add_executable(${test_name} ${test_src})
    # 测试也需要链接分布式库，防止 undefined reference
    target_link_libraries(${test_name} PRIVATE zerollm zerollm_core ${BACKEND_LIBS} ${DISTRIBUTED_LIBS})
    target_compile_definitions(${test_name} PRIVATE ${COMMON_COMPILE_DEFINITIONS})
    add_test(NAME ${test_name}_run COMMAND ${test_name})
endforeach()