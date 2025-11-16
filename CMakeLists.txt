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

set(CMAKE_BUILD_TYPE Debug)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native")
set(CMAKE_CUDA_FLAGS_RELEASE "-O3")

enable_testing()

add_compile_options(
    -fmacro-prefix-map=${CMAKE_SOURCE_DIR}=
)

# -------------------------------
# 选择后端: cuda / rocm / cpu
# -------------------------------
set(ZEROLLM_BACKEND "cuda" CACHE STRING "选择计算后端: cuda / rocm / cpu")
set_property(CACHE ZEROLLM_BACKEND PROPERTY STRINGS cuda rocm cpu)

message(STATUS "选择的后端为: ${ZEROLLM_BACKEND}")

# -------------------------------
# 后端路径映射和预处理器定义
# -------------------------------
if(ZEROLLM_BACKEND STREQUAL "cuda")
    find_package(CUDAToolkit REQUIRED)
    message(STATUS "CUDA Toolkit root: ${CUDAToolkit_ROOT}") 
    include_directories(${CUDAToolkit_INCLUDE_DIRS})
    set(KERNEL_DIR ${PROJECT_SOURCE_DIR}/src/kernel/cuda)
    
    set(BACKEND_LANG CUDA)
    set(BACKEND_LIBS CUDA::cudart)
    set(COMMON_COMPILE_DEFINITIONS USE_CUDA)
elseif(ZEROLLM_BACKEND STREQUAL "rocm")
    find_package(hip REQUIRED)
    message(STATUS "使用 ROCm 后端")
    set(KERNEL_DIR ${PROJECT_SOURCE_DIR}/src/kernel/rocm)
    set(BACKEND_LANG HIP)
    set(BACKEND_LIBS hip::host)
    set(COMMON_COMPILE_DEFINITIONS USE_ROCM)
elseif(ZEROLLM_BACKEND STREQUAL "cpu")
    message(STATUS "使用 CPU 后端")
    set(KERNEL_DIR ${PROJECT_SOURCE_DIR}/src/kernel/cpu)
    set(BACKEND_LANG CXX)
    set(BACKEND_LIBS "")
    set(COMMON_COMPILE_DEFINITIONS USE_CPU)
else()
    message(FATAL_ERROR "无效的 ZEROLLM_BACKEND 选项: ${ZEROLLM_BACKEND}，请使用 cuda / rocm / cpu")
endif()
message(STATUS "User kernel directory: ${KERNEL_DIR}")

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
    ${KERNEL_DIR}
)

# -------------------------------
# 核心算子库（仅包含选定后端）
# -------------------------------
if(ZEROLLM_BACKEND STREQUAL "cuda")
    file(GLOB_RECURSE BACKEND_SOURCES ${KERNEL_DIR}/*.cu)
elseif(ZEROLLM_BACKEND STREQUAL "rocm")
    file(GLOB_RECURSE BACKEND_SOURCES ${KERNEL_DIR}/*.hip)
elseif(ZEROLLM_BACKEND STREQUAL "cpu")
    file(GLOB_RECURSE BACKEND_SOURCES ${KERNEL_DIR}/*.cpp)
endif()

add_library(zerollm_core STATIC ${BACKEND_SOURCES})
set_target_properties(zerollm_core PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
    CUDA_ARCHITECTURES "75;80;86"
)

# 为目标添加预处理器定义
target_compile_definitions(zerollm_core PRIVATE ${COMMON_COMPILE_DEFINITIONS})

# -------------------------------
# 主逻辑部分
# -------------------------------
file(GLOB_RECURSE LOSS_SRCS src/loss/*.cpp)
file(GLOB_RECURSE OPTIMIZER_SRCS src/optimizer/*.cpp)
file(GLOB_RECURSE RUNTIME_SRCS src/runtime/*.cpp)
file(GLOB_RECURSE RUNTIME_TRAINER_SRCS src/runtime/trainer/*.cpp)
file(GLOB_RECURSE MODEL_SRCS src/model/*.cpp)
file(GLOB_RECURSE UTILS_SRCS src/utils/*.cpp)
file(GLOB_RECURSE UTILS_TOKENIZER_SRCS src/utils/tokenizer/*.cpp)
file(GLOB_RECURSE LAYER_SRC src/layer/*.cpp)

set(SOURCES
    src/runtime/trainer/train_zerollm.cpp
    ${OPTIMIZER_SRCS}
    ${RUNTIME_SRCS}
    ${MODEL_SRCS}
    ${UTILS_SRCS}
    ${UTILS_TOKENIZER_SRCS}
    ${LAYER_SRC}
    ${LOSS_SRCS}
)

add_library(zerollm SHARED ${SOURCES})
target_link_libraries(zerollm PRIVATE zerollm_core ${BACKEND_LIBS})
target_compile_definitions(zerollm PRIVATE ${COMMON_COMPILE_DEFINITIONS})


add_executable(train_zerollm src/runtime/trainer/train_zerollm.cpp)
target_link_libraries(train_zerollm PRIVATE zerollm zerollm_core ${BACKEND_LIBS})
target_compile_definitions(train_zerollm PRIVATE ${COMMON_COMPILE_DEFINITIONS})


file(GLOB TOOLS_SOURCES src/tools/*.cpp)
foreach(tool_src ${TOOLS_SOURCES})
    get_filename_component(tool_name ${tool_src} NAME_WE)
    add_executable(${tool_name} ${tool_src})
    target_link_libraries(${tool_name} PRIVATE ${CMAKE_THREAD_LIBS_INIT} rt)
    target_compile_definitions(${tool_name} PRIVATE ${COMMON_COMPILE_DEFINITIONS})
endforeach()


# -------------------------------
# 测试部分
# -------------------------------
file(GLOB TEST_SOURCES tests/*.cpp)
foreach(test_src ${TEST_SOURCES})
    get_filename_component(test_name ${test_src} NAME_WE)
    add_executable(${test_name} ${test_src})
    target_link_libraries(${test_name} PRIVATE zerollm zerollm_core ${BACKEND_LIBS})
    target_compile_definitions(${test_name} PRIVATE ${COMMON_COMPILE_DEFINITIONS})
    add_test(NAME ${test_name}_run COMMAND ${test_name})
endforeach()