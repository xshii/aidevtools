#!/bin/bash
# AI Dev Tools - C++ Golden 编译脚本
#
# 编译所有 C++ 组件:
#   1. cpu_golden     - GFloat 算子 CLI (matmul, softmax, layernorm, transpose)
#   2. gfloat_golden  - GFloat 格式 Python 扩展
#   3. bfp_golden     - BFP 格式 Python 扩展
#
# 用法:
#   ./build_golden.sh          # 编译所有组件
#   ./build_golden.sh cpu      # 仅编译 cpu_golden
#   ./build_golden.sh gfloat   # 仅编译 gfloat_golden
#   ./build_golden.sh bfp      # 仅编译 bfp_golden
#   ./build_golden.sh clean    # 清理编译产物

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="$SCRIPT_DIR/aidevtools"

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 获取 CPU 核心数
get_nproc() {
    nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4
}

# 检查依赖
check_deps() {
    local missing=()

    if ! command -v cmake &> /dev/null; then
        missing+=("cmake")
    fi

    if ! command -v make &> /dev/null; then
        missing+=("make")
    fi

    if ! command -v c++ &> /dev/null && ! command -v g++ &> /dev/null; then
        missing+=("c++ compiler (g++ or clang++)")
    fi

    if [ ${#missing[@]} -ne 0 ]; then
        error "缺少依赖: ${missing[*]}"
        echo "请安装后重试:"
        echo "  macOS:  brew install cmake"
        echo "  Ubuntu: sudo apt install cmake build-essential"
        exit 1
    fi
}

# 激活 venv (如果存在)
activate_venv() {
    if [ -f "$SCRIPT_DIR/.venv/bin/activate" ]; then
        source "$SCRIPT_DIR/.venv/bin/activate"
    fi
}

# 获取 Python 可执行文件路径
get_python_executable() {
    python3 -c "import sys; print(sys.executable)"
}

# 检查 pybind11
check_pybind11() {
    activate_venv
    if ! python3 -c "import pybind11" 2>/dev/null; then
        warn "pybind11 未安装，正在安装..."
        pip install pybind11 -q
    fi
}

# 编译 cpu_golden
build_cpu_golden() {
    info "编译 cpu_golden (GFloat 算子 CLI)..."

    local cpp_dir="$SRC_DIR/golden/cpp"
    local build_dir="$cpp_dir/build"
    local output_dir="$SRC_DIR/golden"

    mkdir -p "$build_dir"
    cd "$build_dir"

    cmake .. -DCMAKE_BUILD_TYPE=Release
    make -j$(get_nproc)

    cp cpu_golden "$output_dir/"
    info "cpu_golden 编译完成: $output_dir/cpu_golden"
}

# 编译 gfloat_golden
build_gfloat_golden() {
    info "编译 gfloat_golden (GFloat Python 扩展)..."

    check_pybind11

    local cpp_dir="$SRC_DIR/formats/custom/gfloat/cpp"
    local build_dir="$cpp_dir/build"
    local output_dir="$SRC_DIR/formats/custom/gfloat"
    local python_exe=$(get_python_executable)

    mkdir -p "$build_dir"
    cd "$build_dir"

    cmake .. -DCMAKE_BUILD_TYPE=Release -DPython3_EXECUTABLE="$python_exe"
    make -j$(get_nproc)

    info "gfloat_golden 编译完成: $output_dir/gfloat_golden*.so"
}

# 编译 bfp_golden
build_bfp_golden() {
    info "编译 bfp_golden (BFP Python 扩展)..."

    check_pybind11

    local cpp_dir="$SRC_DIR/formats/custom/bfp/cpp"
    local build_dir="$cpp_dir/build"
    local output_dir="$SRC_DIR/formats/custom/bfp"
    local python_exe=$(get_python_executable)

    mkdir -p "$build_dir"
    cd "$build_dir"

    cmake .. -DCMAKE_BUILD_TYPE=Release -DPython3_EXECUTABLE="$python_exe"
    make -j$(get_nproc)

    info "bfp_golden 编译完成: $output_dir/bfp_golden*.so"
}

# 清理编译产物
clean() {
    info "清理编译产物..."

    rm -rf "$SRC_DIR/golden/cpp/build"
    rm -f "$SRC_DIR/golden/cpu_golden"

    rm -rf "$SRC_DIR/formats/custom/gfloat/cpp/build"
    rm -f "$SRC_DIR/formats/custom/gfloat/"gfloat_golden*.so

    rm -rf "$SRC_DIR/formats/custom/bfp/cpp/build"
    rm -f "$SRC_DIR/formats/custom/bfp/"bfp_golden*.so

    info "清理完成"
}

# 编译所有
build_all() {
    build_cpu_golden
    echo ""
    build_gfloat_golden
    echo ""
    build_bfp_golden
}

# 显示帮助
show_help() {
    echo "AI Dev Tools - C++ Golden 编译脚本"
    echo ""
    echo "用法: $0 [命令]"
    echo ""
    echo "命令:"
    echo "  all      编译所有组件 (默认)"
    echo "  cpu      仅编译 cpu_golden (GFloat 算子 CLI)"
    echo "  gfloat   仅编译 gfloat_golden (GFloat Python 扩展)"
    echo "  bfp      仅编译 bfp_golden (BFP Python 扩展)"
    echo "  clean    清理编译产物"
    echo "  help     显示帮助"
    echo ""
    echo "组件说明:"
    echo "  cpu_golden     - GFloat 格式的算子 CLI，支持 matmul/softmax/layernorm/transpose"
    echo "  gfloat_golden  - GFloat 格式的 Python 扩展 (pybind11)"
    echo "  bfp_golden     - BFP 块浮点格式的 Python 扩展 (pybind11)"
    echo ""
    echo "示例:"
    echo "  $0              # 编译所有"
    echo "  $0 cpu          # 仅编译 cpu_golden"
    echo "  $0 clean        # 清理"
}

# 主函数
main() {
    echo "=========================================="
    echo "  AI Dev Tools - C++ Golden 编译"
    echo "=========================================="
    echo ""

    check_deps

    case "${1:-all}" in
        all)
            build_all
            ;;
        cpu|cpu_golden)
            build_cpu_golden
            ;;
        gfloat|gfloat_golden)
            build_gfloat_golden
            ;;
        bfp|bfp_golden)
            build_bfp_golden
            ;;
        clean)
            clean
            ;;
        -h|--help|help)
            show_help
            exit 0
            ;;
        *)
            error "未知命令: $1"
            show_help
            exit 1
            ;;
    esac

    echo ""
    echo "=========================================="
    info "编译完成!"
    echo "=========================================="
}

main "$@"
