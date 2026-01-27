# Demos 目录说明

本目录包含 aidevtools 的使用示例。

## 目录结构

```
demos/
├── 01_basic_ops/           # 基础算子示例
├── 02_mini_transformer/    # MiniTransformer 完整比对流程
├── 03_transformer/         # Transformer 模型示例
├── 04_xlsx_basic/          # xlsx 双向工作流示例
├── 05_xlsx_transformer/    # xlsx Transformer 示例
├── 06_add_ops/             # 添加新算子指南
├── 07_transpose/           # Transpose 多维度测试
└── 08_paper_analysis/      # Paper Analysis 时延分析
```

## Demo 说明

### 01_basic_ops - 基础算子

演示 `aidevtools.ops.nn` 中的基础算子用法。

```bash
python demos/01_basic_ops/run.py
```

### 02_mini_transformer - 完整比对流程 (推荐入门)

演示完整的 golden 生成与比对流程：
- 使用 ops API 定义算子序列
- 执行 cpp golden 和 reference
- 三列比对 (exact / fuzzy_pure / fuzzy_qnt)

```bash
python demos/02_mini_transformer/run.py
```

### 03_transformer - Transformer 模型

完整 Transformer 模型示例，展示使用 ops API 构建模型。

```bash
python demos/03_transformer/run.py
```

### 04_xlsx_basic - xlsx 双向工作流

xlsx 配置文件的双向工作流示例：
- Python → Excel: 代码导出为配置
- Excel → Python: 配置生成代码

```bash
python demos/04_xlsx_basic/run.py
```

### 05_xlsx_transformer - xlsx Transformer

从 Excel 配置生成 Transformer 模型。

```bash
python demos/05_xlsx_transformer/run.py
```

### 06_add_ops - 添加新算子指南

以 RMSNorm 为例，说明添加新算子的完整流程。

### 07_transpose - Transpose 测试

测试 Transpose 算子的多维度支持 (2D/3D/4D) 和多种量化类型。

```bash
python demos/07_transpose/run.py
```

### 08_paper_analysis - Paper Analysis 时延分析

演示使用 PyTorch 风格 F API 定义模型，自动生成 OpProfile 进行 Paper Analysis：
- 使用 F.matmul, F.layer_norm, F.softmax, F.gelu 等算子
- 自动收集算子 profile (FLOPs, bytes, arithmetic intensity)
- 基于 Roofline 模型分析时延瓶颈
- 导出 xlsx/csv/json 报告

```bash
python demos/08_paper_analysis/main.py
```

## 支持的算子

| 算子 | 说明 | cpp_golden |
|------|------|------------|
| linear | y = x @ W + b | ✓ |
| matmul | 矩阵乘法 | ✓ |
| softmax | Softmax 激活 | ✓ |
| layernorm | Layer Normalization | ✓ |
| rmsnorm | RMS Normalization (LLaMA) | - |
| transpose | 转置 | ✓ |
| relu | ReLU 激活 | ✓ |
| gelu | GELU 激活 | ✓ |
| silu | SiLU/Swish 激活 (LLaMA) | ✓ |
| sigmoid | Sigmoid 激活 | ✓ |
| tanh | Tanh 激活 | ✓ |
| add/mul/div | 逐元素运算 | ✓ |
| attention | Scaled Dot-Product Attention | - |
| embedding | Token 嵌入 | - |

## 量化格式

| 格式 | mantissa | block_size | 用途 |
|------|----------|------------|------|
| bfp4 | 2-bit | 64 | 极端量化 |
| bfp8 | 4-bit | 32 | 通用 |
| bfp16 | 8-bit | 16 | 高精度 |
| gfloat4 | 1+2+1 | - | 实验性 |
| gfloat8 | 1+4+3 | - | 低精度 |
| gfloat16 | 1+8+7 | - | 接近 fp16 |

## 运行要求

```bash
# 安装依赖
pip install numpy openpyxl

# 或使用 install.sh
./install.sh dev
source .venv/bin/activate
```
