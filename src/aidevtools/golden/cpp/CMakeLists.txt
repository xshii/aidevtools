cmake_minimum_required(VERSION 3.10)
project(cpu_golden CXX)

# C++11 标准
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 编译选项
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -O2")

# 可执行文件
add_executable(cpu_golden
    main.cpp
    gfloat_ops.cpp
)

# 安装目标 (可选)
install(TARGETS cpu_golden DESTINATION bin)
