cmake_minimum_required(VERSION 3.10)
project(cpu_golden CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -O2")

# 包含目录
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

# ============================================================
# 选择数据格式: gfloat 或 bfp
# ============================================================
# 默认使用 gfloat，切换到 bfp 只需改这一行
set(FORMAT "gfloat")
# set(FORMAT "bfp")

# 源文件
set(SOURCES
    main.cpp
    ${FORMAT}/io.cpp
    ${FORMAT}/impl.cpp
)

add_executable(cpu_golden ${SOURCES})

install(TARGETS cpu_golden DESTINATION bin)

# ============================================================
# 目录结构说明
# ============================================================
#
# cpp/
# ├── gfloat/          # GFloat 格式全套
# │   ├── io.h/cpp     # I/O + 格式转换
# │   └── impl.cpp     # 算子实现
# ├── bfp/             # BFP 格式全套
# │   ├── io.h/cpp     # I/O + 格式转换
# │   └── impl.cpp     # 算子实现
# ├── interface.h      # 通用算子接口
# └── main.cpp         # CLI 框架
#
# 切换格式: 修改 set(FORMAT "gfloat") 为 set(FORMAT "bfp")
# 替换算子: 替换 ${FORMAT}/impl.cpp
