#!/bin/bash
# GFloat CPU Golden 编译脚本

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="$SCRIPT_DIR/build"

echo "=== Building cpu_golden ==="
echo "Source: $SCRIPT_DIR"
echo "Build:  $BUILD_DIR"

# 创建 build 目录
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

# 运行 CMake
cmake ..

# 编译
make -j$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)

# 复制到 golden 目录
cp cpu_golden "$SCRIPT_DIR/../"

echo ""
echo "=== Build complete ==="
echo "Executable: $SCRIPT_DIR/../cpu_golden"
echo ""
echo "Usage:"
echo "  ./cpu_golden matmul gfp16 a.bin b.bin c.bin 64 128 256"
echo "  ./cpu_golden softmax gfp8 input.bin output.bin 4 64"
echo "  ./cpu_golden layernorm gfp16 x.bin gamma.bin beta.bin y.bin 4 256"
