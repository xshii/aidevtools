#!/bin/bash
# CPU Golden - 统一入口点
# 根据 dtype 参数自动路由到 GFloat 或 BFP 实现
#
# 用法:
#   cpu_golden <op> <dtype> ...
#
# 示例:
#   cpu_golden matmul gfp16 a.bin b.bin c.bin 64 128 256   # -> cpu_golden_gfloat
#   cpu_golden matmul bfp16 a.bin b.bin c.bin 64 128 256   # -> cpu_golden_bfp

# 获取脚本所在目录（处理软链接）
if [[ -L "$0" ]]; then
    SCRIPT_DIR="$(cd "$(dirname "$(readlink "$0")")" && pwd)"
else
    SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
fi

# 处理特殊参数（--help, -h, help）
if [ $# -eq 0 ] || [[ "$1" == "-h" ]] || [[ "$1" == "--help" ]] || [[ "$1" == "help" ]]; then
    # 默认路由到 GFloat backend 显示帮助
    BACKEND="$SCRIPT_DIR/cpu_golden_gfloat"
    if [ -x "$BACKEND" ]; then
        exec "$BACKEND" "$@"
    else
        echo "Error: backend not found: $BACKEND" >&2
        exit 1
    fi
fi

# 检查参数数量
if [ $# -lt 2 ]; then
    echo "Error: insufficient arguments" >&2
    echo "Usage: $0 <op> <dtype> ..." >&2
    echo "" >&2
    echo "Supported dtypes:" >&2
    echo "  GFloat: gfp4, gfp8, gfp16 (or gfloat4, gfloat8, gfloat16)" >&2
    echo "  BFP:    bfp4, bfp8, bfp16" >&2
    exit 1
fi

# 提取 dtype 参数（可能在第2或第3个位置）
# 对于 matmul_mixed: <op> <dtype_a> <dtype_b> ...
# 对于其他命令: <op> <dtype> ...
DTYPE="$2"

# 判断格式类型并路由
if [[ "$DTYPE" =~ ^(bfp|BFP) ]]; then
    # BFP 格式
    BACKEND="$SCRIPT_DIR/cpu_golden_bfp"
else
    # GFloat 格式（默认）
    BACKEND="$SCRIPT_DIR/cpu_golden_gfloat"
fi

# 检查后端是否存在
if [ ! -x "$BACKEND" ]; then
    echo "Error: backend not found or not executable: $BACKEND" >&2
    exit 1
fi

# 路由到对应的实现
exec "$BACKEND" "$@"
