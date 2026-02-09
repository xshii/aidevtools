#!/bin/bash
# Compare Demo 批量运行脚本
# 运行所有 7 个 compare demos

set -e  # 遇到错误就退出

echo "============================================================================"
echo "  运行所有 Compare Demos"
echo "  日期: $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================================"
echo ""

DEMOS=(
    "compare_01_encoder_pure_vs_fp16"
    "compare_02_encoder_pure_vs_fuzzq"
    "compare_03_encoder_fuzzq_vs_qa"
    "compare_04_matmul_pure_vs_dut"
    "compare_06_encoder_bfp8_fuzzq"
    "compare_07_encoder_bfp8_qa"
    "compare_08_transformer_bfp8_qa"
)

PASSED=0
FAILED=0
FAILED_DEMOS=()

for demo in "${DEMOS[@]}"; do
    echo "============================================================================"
    echo "  运行: $demo"
    echo "============================================================================"

    if python "demos/compare/$demo/run.py"; then
        echo "✅ $demo 通过"
        ((PASSED++))
    else
        echo "❌ $demo 失败"
        ((FAILED++))
        FAILED_DEMOS+=("$demo")
    fi
    echo ""
done

echo "============================================================================"
echo "  运行总结"
echo "============================================================================"
echo "  通过: $PASSED / ${#DEMOS[@]}"
echo "  失败: $FAILED / ${#DEMOS[@]}"

if [ $FAILED -gt 0 ]; then
    echo ""
    echo "  失败的 demos:"
    for demo in "${FAILED_DEMOS[@]}"; do
        echo "    - $demo"
    done
    exit 1
else
    echo ""
    echo "  ✅ 所有 demos 运行成功！"
    exit 0
fi
