#!/bin/bash
# AI Dev Tools 安装脚本

VENV_DIR=".venv"

echo "=== AI Dev Tools 安装 ==="

# 创建虚拟环境
if [ ! -d "$VENV_DIR" ]; then
    echo "创建虚拟环境..."
    python3 -m venv "$VENV_DIR"
fi

# 激活并安装
source "$VENV_DIR/bin/activate"
pip install --upgrade pip -q

# 根据参数安装
case "${1:-base}" in
    base)
        echo "安装基础包..."
        pip install -e .
        ;;
    dev)
        echo "安装开发环境..."
        pip install -e ".[dev]"
        ;;
    vscode)
        echo "安装 VS Code 支持..."
        pip install -e ".[vscode]"
        ;;
    all)
        echo "安装全部依赖..."
        pip install -e ".[dev,vscode]"
        ;;
    *)
        echo "用法: $0 [base|dev|vscode|all]"
        exit 1
        ;;
esac

# 编译 C++ Golden (如果有 cmake)
if command -v cmake &> /dev/null; then
    echo ""
    echo "编译 C++ Golden..."
    ./build_golden.sh all
else
    echo ""
    echo "提示: 未检测到 cmake，跳过 C++ Golden 编译"
    echo "      如需 C++ 加速，请安装 cmake 后运行: ./build_golden.sh"
fi

echo ""
echo "安装完成！"
echo "激活环境: source $VENV_DIR/bin/activate"
echo "运行: aidev"
