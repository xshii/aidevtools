# AI Dev Tools

AI 算子开发工具集，用于自研芯片算子的 Golden 生成与精度验证。

## 特性

- **统一 Tensor 格式**：同时包含 fp32 (最高精度) 和 quantized (量化) 数据
- **双轨 Golden 验证**：pure (纯 fp32) / quant (量化感知) 两种精度模式
- **三列比对机制**：exact + fuzzy_pure + fuzzy_qnt，精确定位误差来源
- **状态判定**：PERFECT → PASS → QUANT_ISSUE → FAIL
- **多种量化格式**：BFP (块浮点)、GFloat、float16 等
- **Python/C++ 双模式**：Golden 算子同时支持 Python 和 C++ 实现

## 快速上手

```bash
# 1. 安装
./install.sh dev

# 2. 激活环境
source .venv/bin/activate

# 3. 编译 C++ Golden (可选，启用 C++ 加速)
./build_golden.sh

# 4. 运行 Demo
python demos/02_mini_transformer/run.py

# 5. 运行测试
pytest tests/ -v
```

## 基础使用

```python
from aidevtools.core import (
    set_config, get_config,
    Tensor, generate_random, generate_weight,
    get_engine, clear, list_ops,
)
from aidevtools.tools.compare.diff import compare_3col, print_compare_table

# 1. 配置全局参数
set_config(
    golden_mode="python",  # python | cpp
    precision="quant",     # pure | quant
    seed=42,
)

# 2. 生成测试数据
x = generate_random(shape=(2, 4, 64), qtype="bfp8", seed=42)
w = generate_weight(shape=(64, 128), qtype="bfp8", seed=43)

# 3. 执行算子
engine = get_engine()
y = engine.run_op("linear", inputs=[x], weights=[w], qtype="bfp8")

# 4. 获取 Golden 结果
for r in engine.get_records():
    print(f"{r.op_name}: golden_pure={r.golden_pure.shape}, golden_quant={r.golden_quant.shape}")
```

## 架构概览

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           应用层 (Application)                           │
│   demos/  │  xlsx 工作流  │  Python API  │  CLI                         │
└─────────────────────────────────┬───────────────────────────────────────┘
                                  │
┌─────────────────────────────────▼───────────────────────────────────────┐
│                            核心层 (Core)                                 │
│   config (全局配置)  │  tensor (统一张量)  │  op (算子)  │  engine (引擎) │
└─────────────────────────────────┬───────────────────────────────────────┘
                                  │
┌─────────────────────────────────▼───────────────────────────────────────┐
│                            工具层 (Tools)                                │
│   compare (三列比对)  │  trace (插桩)  │  xlsx (表格)  │  archive (打包) │
└─────────────────────────────────┬───────────────────────────────────────┘
                                  │
┌─────────────────────────────────▼───────────────────────────────────────┐
│                           格式层 (Formats)                               │
│   numpy (npy/npz)  │  raw (二进制)  │  bfp (块浮点)  │  gfloat (自定义)  │
└─────────────────────────────────────────────────────────────────────────┘
```

## 三列比对

| 列 | 说明 |
|----|------|
| **exact** | 精确比对 (bit-level 或阈值内) |
| **fuzzy_pure** | 模糊比对 vs 纯 fp32 Golden |
| **fuzzy_qnt** | 模糊比对 vs 量化感知 Golden |

### 状态判定

| 状态 | 条件 | 含义 |
|------|------|------|
| PERFECT | exact ✓ | 完全一致 |
| PASS | fuzzy_qnt ✓ | 误差在容差内 |
| QUANT_ISSUE | fuzzy_pure ✓ 且 fuzzy_qnt ✗ | 量化引入的误差 |
| FAIL | 都 ✗ | 算法实现错误 |

### 输出示例

```
====================================================================================================
op_name          exact  f_pure   f_qnt     max_abs     qsnr   cosine    status
----------------------------------------------------------------------------------------------------
linear_0           ✓       ✓       ✓      0.00e+00      inf 1.000000   PERFECT
relu_0             ✗       ✓       ✓      9.54e-07    129.2 1.000000     PASS
softmax_0          ✗       ✓       ✗      1.00e-01     29.0 0.999896 QUANT_ISSUE
matmul_0           ✗       ✗       ✗      1.00e+00      8.8 0.993808     FAIL
====================================================================================================
Summary: 1 PERFECT, 1 PASS, 1 QUANT_ISSUE, 1 FAIL (total: 4)
```

## 支持的量化格式

| 格式 | 说明 | 存储类型 |
|------|------|----------|
| float32 | 原始精度 | float32 |
| float16 | 半精度 | float16 |
| bfp16 | 块浮点 (8-bit mantissa) | int8 + shared_exp |
| bfp8 | 块浮点 (4-bit mantissa) | int8 + shared_exp |
| bfp4 | 块浮点 (2-bit mantissa) | int8 + shared_exp |
| gfloat16 | 自定义 16 位 (1+8+7) | uint16 |
| gfloat8 | 自定义 8 位 (1+4+3) | uint8 |
| gfloat4 | 自定义 4 位 (1+2+1) | uint8 |

## 内置算子

| 类型 | 算子 | cpp_golden | 说明 |
|------|------|:----------:|------|
| **线性变换** | linear | ✓ | y = x @ W + b |
| | matmul | ✓ | 矩阵乘法 (支持混合精度) |
| | transpose | ✓ | 转置 (2D/3D/4D) |
| **激活函数** | relu | - | ReLU |
| | gelu | - | GELU |
| | silu | - | SiLU/Swish (LLaMA FFN) |
| | sigmoid | - | Sigmoid |
| | tanh | - | Tanh |
| | softmax | ✓ | Softmax |
| **归一化** | layernorm | ✓ | Layer Normalization |
| | rmsnorm | - | RMS Normalization (LLaMA/Mistral) |
| | batchnorm | - | Batch Normalization |
| **注意力** | attention | - | Scaled Dot-Product Attention |
| **元素运算** | add, mul, div | - | 逐元素运算 |
| **嵌入** | embedding | - | Token 嵌入 |

## 目录结构

```
aidevtools/
├── src/aidevtools/
│   ├── core/               # 核心: config, tensor, op, engine
│   ├── formats/            # 格式: numpy, raw, bfp, gfloat
│   ├── tools/compare/      # 比对: diff, report, export
│   ├── trace/              # 插桩
│   └── xlsx/               # Excel 工作流
├── demos/                  # 示例脚本
├── tests/                  # 测试用例
└── docs/                   # 文档
    └── architecture.md     # 架构设计
```

## 文档

- [架构设计](docs/architecture.md) - 详细架构说明
- [比数指南](docs/compare_guide.md) - 比对工具使用
- [BFP 格式](src/aidevtools/formats/custom/bfp/guide.md) - 块浮点格式
- [GFloat 格式](src/aidevtools/formats/custom/gfloat/guide.md) - 自定义浮点格式

## 开发

```bash
# 安装开发依赖
./install.sh dev

# 编译 C++ Golden
./build_golden.sh          # 编译所有
./build_golden.sh cpu      # 仅编译 cpu_golden CLI
./build_golden.sh gfloat   # 仅编译 gfloat Python 扩展
./build_golden.sh bfp      # 仅编译 bfp Python 扩展
./build_golden.sh clean    # 清理编译产物

# 运行测试
pytest tests/ -v

# 运行覆盖率
pytest tests/ --cov=aidevtools --cov-report=term-missing

# 一键 CI
./ci.sh
```

### C++ Golden 组件

| 组件 | 类型 | 说明 |
|------|------|------|
| cpu_golden | CLI | GFloat 格式算子命令行工具 |
| gfloat_golden | Python 扩展 | GFloat 格式量化/反量化 |
| bfp_golden | Python 扩展 | BFP 块浮点格式量化/反量化 |

## 环境要求

- Python >= 3.8
- numpy
- openpyxl (xlsx 工作流，可选)
- pybind11 (C++ 扩展，可选)
- cmake >= 3.14 (编译 C++ Golden，可选)

## License

MIT
