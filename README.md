# AI Dev Tools

AI 算子开发工具集，用于自研芯片算子的 Golden 生成与精度验证。

## 特性

- **统一 Tensor 格式**：同时包含 fp32 (最高精度) 和 quantized (量化) 数据
- **双轨 Golden 验证**：pure (纯 fp32) / quant (量化感知) 两种精度模式
- **四状态判定**：PASS / GOLDEN_SUSPECT / DUT_ISSUE / BOTH_SUSPECT
- **Golden 自检**：自动检测 Golden 数据有效性（非零、无 NaN/Inf、QSNR 阈值）
- **多种量化格式**：BFP (块浮点)、GFloat、float16 等
- **Python/C++ 双模式**：Golden 算子同时支持 Python 和 C++ 实现

## 快速上手

```bash
# 1. 安装 (自动编译 C++ Golden)
./install.sh dev

# 2. 激活环境
source .venv/bin/activate

# 3. 运行 Demo
python demos/02_mini_transformer/run.py

# 4. 运行测试
pytest tests/ -v
```

> 注：安装时如检测到 cmake，会自动编译 C++ Golden。如需手动编译，可运行 `./build_golden.sh`。

## 基础使用

### PyTorch 风格 API

```python
import numpy as np
from aidevtools import ops
from aidevtools.ops import _functional as F

# 清空记录
ops.clear()

# 使用 F API 执行算子
x = np.random.randn(2, 8, 64).astype(np.float32)
w = np.random.randn(64, 128).astype(np.float32)

y = F.matmul(x, w)           # 矩阵乘法
y = F.layer_norm(y, (128,))  # LayerNorm
y = F.softmax(y, dim=-1)     # Softmax

# 获取 Golden 记录
for r in ops.get_records():
    print(f"{r.op_name}: input={r.input.shape}, golden={r.golden.shape}")
```

### 比对 API

```python
from aidevtools.compare import CompareEngine, CompareConfig

# 创建比对引擎
config = CompareConfig(
    fuzzy_min_qsnr=30.0,
    fuzzy_min_cosine=0.999,
)
engine = CompareEngine(config)

# 执行比对
result = engine.compare(
    dut_output=dut,           # DUT 输出
    golden_pure=golden_fp32,  # 纯 fp32 Golden
    golden_qnt=golden_qnt,    # 量化感知 Golden
)
print(f"Status: {result.status.value}")
```

### 数据生成 API

```python
from aidevtools.frontend import DataGenerator, DType

gen = DataGenerator(seed=42)

# 生成输入数据
x = gen.gen_input(shape=(2, 64), dtype="bfp16", dist="normal")

# 生成权重数据
w = gen.gen_weight(shape=(64, 128), dtype="bfp16", init="xavier")
```

## 架构概览

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           应用层 (Application)                           │
│   demos/  │  xlsx 工作流  │  Python API  │  CLI                         │
└─────────────────────────────────┬───────────────────────────────────────┘
                                  │
┌─────────────────────────────────▼───────────────────────────────────────┐
│                            前端层 (Frontend)                             │
│   types (统一类型)  │  datagen (数据生成)  │  compile (编译封装)          │
└─────────────────────────────────┬───────────────────────────────────────┘
                                  │
┌─────────────────────────────────▼───────────────────────────────────────┐
│                            比对层 (Compare)                              │
│   exact (精确)  │  fuzzy (模糊)  │  sanity (自检)  │  engine (引擎)       │
└─────────────────────────────────┬───────────────────────────────────────┘
                                  │
┌─────────────────────────────────▼───────────────────────────────────────┐
│                           格式层 (Formats)                               │
│   numpy (npy/npz)  │  raw (二进制)  │  bfp (块浮点)  │  gfloat (自定义)  │
└─────────────────────────────────────────────────────────────────────────┘
```

## 四状态判定模型

| DUT vs Golden | Golden 自检 | 判定状态 | 含义 |
|---------------|-------------|----------|------|
| PASS | PASS | **PASS** | DUT 正确，Golden 有效 |
| PASS | FAIL | **GOLDEN_SUSPECT** | DUT 匹配，但 Golden 可疑 |
| FAIL | PASS | **DUT_ISSUE** | Golden 有效，DUT 有问题 |
| FAIL | FAIL | **BOTH_SUSPECT** | 都可疑，需人工排查 |

### Golden 自检项

- **non_zero**: 数据非全零
- **no_nan_inf**: 无 NaN/Inf
- **range_valid**: 数值范围合理
- **qsnr_valid**: golden_qnt vs golden_pure QSNR >= 阈值

### 输出示例

```
==============================================================================================================
name            exact  f_pure   f_qnt   sanity     max_abs     qsnr   cosine        status
--------------------------------------------------------------------------------------------------------------
matmul_0           Y       Y       Y       Y     0.00e+00      inf 1.000000          PASS
layernorm_0        N       Y       Y       Y     2.52e-01    17.54 0.991358          PASS
softmax_0          N       N       N       N     2.63e-02    14.54 0.982997   BOTH_SUSPECT
==============================================================================================================
Summary: 2 PASS, 0 GOLDEN_SUSPECT, 0 DUT_ISSUE, 1 BOTH_SUSPECT (total: 3)
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
| **激活函数** | relu | ✓ | ReLU |
| | gelu | ✓ | GELU |
| | silu | ✓ | SiLU/Swish (LLaMA FFN) |
| | sigmoid | ✓ | Sigmoid |
| | tanh | ✓ | Tanh |
| | softmax | ✓ | Softmax |
| **归一化** | layernorm | ✓ | Layer Normalization |
| | rmsnorm | - | RMS Normalization (LLaMA/Mistral) |
| | batchnorm | - | Batch Normalization |
| **注意力** | attention | - | Scaled Dot-Product Attention |
| **元素运算** | add, mul, div | ✓ | 逐元素运算 |
| **嵌入** | embedding | - | Token 嵌入 |

## 目录结构

```
aidevtools/
├── aidevtools/
│   ├── core/               # 核心: config, tensor, log
│   ├── compare/            # 比对: exact, fuzzy, sanity, engine, report
│   ├── frontend/           # 前端: types, datagen, compile
│   ├── formats/            # 格式: numpy, raw, bfp, gfloat
│   ├── golden/             # Golden 实现
│   │   ├── cpu_golden      # 编译后的 CLI 可执行文件
│   │   └── cpp/            # C++ Golden 源码
│   ├── ops/                # 算子: functional API, cpu_golden
│   ├── tools/compare/      # 工具: diff, report, export
│   ├── trace/              # 插桩
│   └── xlsx/               # Excel 工作流
├── demos/                  # 示例脚本
│   ├── 01_basic_ops/       # 基础算子示例
│   ├── 02_mini_transformer/# Mini Transformer
│   ├── 03_transformer/     # 完整 Transformer
│   ├── 04_xlsx_basic/      # xlsx 基础工作流
│   ├── 05_xlsx_transformer/# xlsx Transformer
│   ├── 07_transpose/       # Transpose 示例
│   └── 08_paper_analysis/  # Paper Analysis
├── tests/                  # 测试用例
└── docs/                   # 文档
```

## 文档

- [架构设计](docs/architecture.md) - 详细架构说明
- [比数指南](docs/compare_guide.md) - 比对工具使用
- [BFP 格式](aidevtools/formats/custom/bfp/guide.md) - 块浮点格式
- [GFloat 格式](aidevtools/formats/custom/gfloat/guide.md) - 自定义浮点格式

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

# 代码检查
ruff check aidevtools/ tests/
pylint aidevtools/

# 一键 CI
./ci.sh
```

### C++ Golden 组件

| 组件 | 类型 | 说明 |
|------|------|------|
| cpu_golden | CLI | 算子命令行工具 (matmul/softmax/layernorm/transpose 等) |
| gfloat_golden | Python 扩展 | GFloat 格式量化/反量化 |
| bfp_golden | Python 扩展 | BFP 块浮点格式量化/反量化 |

### 模块说明

| 模块 | 说明 |
|------|------|
| `aidevtools.compare` | 比对引擎，支持精确/模糊比对、Golden 自检、报告生成 |
| `aidevtools.frontend` | 前端统一 API，数据生成、编译封装 |
| `aidevtools.ops` | 算子实现，PyTorch 风格 F API |
| `aidevtools.formats` | 数据格式，BFP/GFloat 量化 |

## 环境要求

- Python >= 3.8
- numpy
- openpyxl (xlsx 工作流，可选)
- pybind11 (C++ 扩展，可选)
- cmake >= 3.14 (编译 C++ Golden，可选)

## License

MIT
