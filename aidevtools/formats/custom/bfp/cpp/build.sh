#!/bin/bash
# BFP Golden API 编译脚本

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/build"
OUTPUT_DIR="${SCRIPT_DIR}/.."

echo "=== BFP Golden API 编译 ==="

# 检查 pybind11
if ! python3 -c "import pybind11" 2>/dev/null; then
    echo "安装 pybind11..."
    pip3 install pybind11 || pip install pybind11
fi

# 方式 1: 使用 CMake
if command -v cmake &> /dev/null; then
    echo "使用 CMake 编译..."
    mkdir -p "${BUILD_DIR}"
    cd "${BUILD_DIR}"
    cmake ..
    make -j$(nproc 2>/dev/null || sysctl -n hw.ncpu)
    echo "编译完成: ${OUTPUT_DIR}/bfp_golden*.so"
    exit 0
fi

# 方式 2: 直接使用编译器
echo "使用编译器直接编译..."
cd "${SCRIPT_DIR}"

PYBIND11_INCLUDES=$(python3 -m pybind11 --includes)
PYTHON_EXT_SUFFIX=$(python3-config --extension-suffix)

if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS
    c++ -O3 -Wall -shared -std=c++17 -undefined dynamic_lookup \
        ${PYBIND11_INCLUDES} \
        bfp.cpp bfp_pybind.cpp \
        -o "${OUTPUT_DIR}/bfp_golden${PYTHON_EXT_SUFFIX}"
else
    # Linux
    c++ -O3 -Wall -shared -std=c++17 -fPIC \
        ${PYBIND11_INCLUDES} \
        bfp.cpp bfp_pybind.cpp \
        -o "${OUTPUT_DIR}/bfp_golden${PYTHON_EXT_SUFFIX}"
fi

echo "编译完成: ${OUTPUT_DIR}/bfp_golden${PYTHON_EXT_SUFFIX}"
