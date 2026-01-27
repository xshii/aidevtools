cmake_minimum_required(VERSION 3.10)
project(cpu_golden CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -O2")

# 包含目录
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

# ============================================================
# 构建两个可执行文件: cpu_golden (gfloat) 和 cpu_golden_bfp
# ============================================================

# GFloat 版本 (默认)
set(GFLOAT_SOURCES
    main.cpp
    gfloat/io.cpp
    gfloat/impl.cpp
)
add_executable(cpu_golden ${GFLOAT_SOURCES})

# BFP 版本
set(BFP_SOURCES
    bfp_main.cpp
    bfp/io.cpp
    bfp/impl.cpp
)
add_executable(cpu_golden_bfp ${BFP_SOURCES})

install(TARGETS cpu_golden cpu_golden_bfp DESTINATION bin)

# ============================================================
# 目录结构说明
# ============================================================
#
# cpp/
# ├── gfloat/          # GFloat 格式全套
# │   ├── io.h/cpp     # I/O + 格式转换
# │   └── impl.cpp     # 算子实现 (含 _gfloat 精度模拟版本)
# ├── bfp/             # BFP 格式全套
# │   ├── io.h/cpp     # I/O + 格式转换
# │   └── impl.cpp     # 算子实现 (含 _bfp 精度模拟版本)
# ├── interface.h      # 通用算子接口 (fp32 + gfloat + bfp 声明)
# ├── main.cpp         # GFloat CLI
# └── bfp_main.cpp     # BFP CLI
#
# 构建输出:
#   - cpu_golden:     使用 gfloat 格式 (gfp4/gfp8/gfp16)
#   - cpu_golden_bfp: 使用 bfp 格式 (bfp4/bfp8/bfp16)
