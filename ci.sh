#!/bin/bash
# 一键本地 CI

set -e

echo "=== AI Dev Tools CI ==="

# 检查虚拟环境
if [ ! -d ".venv" ]; then
    echo "创建虚拟环境..."
    python3 -m venv .venv
fi

source .venv/bin/activate
pip install -q -e ".[dev]"

echo ""
echo "[1/3] Lint 检查..."
ruff check aidevtools/ tests/ demos/ || true

echo ""
echo "[2/3] 运行测试..."
pytest tests/ -v --tb=short

echo ""
echo "[3/3] 覆盖率..."
pytest tests/ --cov=aidevtools --cov-report=term-missing --cov-report=html

echo ""
echo "=== CI 完成 ==="
echo "覆盖率报告: htmlcov/index.html"
