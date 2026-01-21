cmake_minimum_required(VERSION 3.10)
project(cpu_golden CXX)

# C++11 标准
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 编译选项
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -O2")

# 源文件
# 架构说明:
#   - gfloat_io.cpp:  GFloat 格式转换与文件 I/O (可替换)
#   - ops_impl.cpp:   算子实现 (可替换为你自己的实现)
#   - main.cpp:       CLI 框架

set(SOURCES
    main.cpp
    gfloat_io.cpp
    ops_impl.cpp
)

# 可执行文件
add_executable(cpu_golden ${SOURCES})

# 安装目标 (可选)
install(TARGETS cpu_golden DESTINATION bin)

# ============================================================
# 自定义编译说明
# ============================================================
#
# 如果你想替换算子实现:
#   1. 保留: main.cpp, gfloat_io.cpp, gfloat_io.h, ops_interface.h
#   2. 替换: ops_impl.cpp -> 你的实现文件
#   3. 修改上面的 SOURCES 列表
#
# 如果你想使用 BFP 格式:
#   1. 创建 bfp_io.h/cpp (参考 gfloat_io 的接口)
#   2. 修改 main.cpp 使用你的 bfp_io
#   3. 或者创建单独的 main_bfp.cpp
#
# 示例 (替换算子实现):
#   set(SOURCES
#       main.cpp
#       gfloat_io.cpp
#       my_ops_impl.cpp  # 你的实现
#   )
