cmake_minimum_required(VERSION 3.10)
project(cpu_golden CXX)

# C++11 标准
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 编译选项
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -O2")

# 包含目录
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

# 源文件
# 目录结构:
#   io/     - 格式 I/O (gfloat_io, bfp_io)
#   ops/    - 算子 (interface.h, impl.cpp)
#   main.cpp - CLI 框架

set(SOURCES
    main.cpp
    io/gfloat_io.cpp
    ops/impl.cpp
)

# 可执行文件
add_executable(cpu_golden ${SOURCES})

# 安装目标 (可选)
install(TARGETS cpu_golden DESTINATION bin)

# ============================================================
# 自定义编译说明
# ============================================================
#
# 目录结构:
#   io/          - 格式 I/O 层 (可替换)
#   ops/         - 算子层 (可替换 impl.cpp)
#   main.cpp     - CLI 框架
#
# 替换算子实现:
#   set(SOURCES
#       main.cpp
#       io/gfloat_io.cpp
#       ops/my_impl.cpp      # 你的实现
#   )
#
# 使用 BFP 格式:
#   1. 修改 main.cpp 使用 io/bfp_io.h
#   2. 添加 io/bfp_io.cpp 到 SOURCES
