# Demos 目录说明

本目录包含 aidevtools 的使用示例，每个示例独立放置在一个文件夹中。

## 目录结构

```
demos/
├── README.md                    # 本文件
├── 01_basic_ops/                # 基础算子示例
│   ├── README.md
│   └── run.py
├── 02_unified_workflow/         # 统一工作流示例
│   ├── README.md
│   └── run.py
├── 03_transformer/              # Transformer 模型示例
│   ├── README.md
│   ├── operators.py             # 算子导入
│   ├── model.py                 # 模型定义 + BFP 量化
│   └── run.py                   # 运行入口
├── 04_xlsx_basic/               # xlsx 双向工作流示例
│   ├── README.md
│   ├── mlp_config.xlsx          # MLP 配置 (4 算子)
│   └── run.py
└── 05_xlsx_transformer/         # xlsx Transformer 示例
    ├── README.md
    ├── transformer_config.xlsx  # Transformer 配置 (13 算子)
    └── run.py
```

## Demo 说明

### 01_basic_ops - 基础算子

演示 `aidevtools.ops.nn` 中的基础算子用法。

```bash
cd demos/01_basic_ops
python run.py
```

包含算子：linear, relu, gelu, softmax, layernorm, attention, embedding

### 02_unified_workflow - 统一工作流 (推荐入门)

演示新架构的完整流程：
- 全局配置 (golden_mode, precision, seed)
- 统一 Tensor 格式 (fp32 + quantized)
- 三列比对 (exact, fuzzy_pure, fuzzy_qnt)
- 状态判定 (PERFECT → PASS → QUANT_ISSUE → FAIL)

```bash
cd demos/02_unified_workflow
python run.py
```

### 03_transformer - Transformer 模型

完整 Transformer 模型示例，展示实际项目的组织方式：
- 算子使用核心库 `aidevtools.ops.nn`
- 模型组合 + BFP 量化策略
- 量化策略：matmul=bfp4, 其他=bfp8

```bash
cd demos/03_transformer
python run.py
```

### 04_xlsx_basic - xlsx 双向工作流

xlsx 配置文件的双向工作流示例：
- Python → Excel: 代码导出为配置
- Excel → Python: 配置生成代码

```bash
cd demos/04_xlsx_basic
python run.py
```

### 05_xlsx_transformer - xlsx Transformer

从 Excel 配置生成 Transformer 模型：
- 在 xlsx 中定义算子序列
- 自动生成 Python 代码
- 运行并比对结果

```bash
cd demos/05_xlsx_transformer
python run.py
```

## 量化格式

| 格式 | mantissa | block_size | 用途 |
|------|----------|------------|------|
| bfp4 | 2-bit | 64 | matmul (极端量化) |
| bfp8 | 4-bit | 32 | 激活函数、LayerNorm |
| bfp16 | 8-bit | 16 | 高精度场景 |
| gfloat4 | 1+2+1 | - | 实验性 |
| gfloat8 | 1+4+3 | - | 通用低精度 |
| gfloat16 | 1+8+7 | - | 接近 fp16 |

## 运行要求

```bash
# 安装依赖
pip install numpy openpyxl

# 或使用 install.sh
./install.sh dev
source .venv/bin/activate
```
