# Demos 目录说明

本目录包含 aidevtools 的使用示例，按功能分为五大类。

## 目录结构

```
demos/
│
├── [入门] 基础算子与模型
│   ├── 01_basic_ops/               # 基础算子 (PyTorch 劫持模式)
│   ├── 02_mini_transformer/        # MiniTransformer 完整比对流程 (推荐入门)
│   └── 03_transformer/             # 完整 Transformer 模型
│
├── [数据生成] 五种输入方式 — 同一 Encoder, 五种写法, bfp8 精度
│   └── datagen/
│       ├── 00_datagen_manual/      # 方式 1: DataGenerator 手动 API
│       ├── 01_datagen_autogen/     # 方式 2: 算子自动生成 API
│       ├── 02_model_dsl/           # 方式 3: Model DSL (纯 Python, 无需编译器)
│       ├── 03_torch_golden/        # 方式 4: PyTorch 劫持 (Golden Mode)
│       └── 04_xlsx_config/         # 方式 5: Excel/XLSX 配置
│
├── [比数] 量化比对与报告
│   └── compare/
│       ├── 06_encoder_bfp8/        # Encoder 全局 bfp8 四种比数 (三种前端模式)
│       └── 07_qa_encoder_bfp8/     # 量化感知随机 vs 正态 bfp8 比数
│
├── [优化器] 时延分析与融合评估
│   └── optimizer/
│       ├── 01_basic/               # 基础用法: PyTorch 劫持 → Benchmark → 评估
│       ├── 02_calibration/         # ML 校准流程
│       ├── 03_fusion_rules/        # 融合规则配置
│       ├── 04_echarts/             # ECharts 可视化
│       ├── 05_comparison/          # 理论 vs 工程化对比
│       └── 06_bridge/              # PyTorch → Benchmark 桥接
│
├── [工具] 开发与分析
│   ├── 04_xlsx_basic/              # xlsx 双向工作流
│   ├── 05_xlsx_transformer/        # xlsx Transformer
│   ├── 06_add_ops/                 # 添加新算子指南
│   ├── 07_transpose/              # Transpose 多维度测试
│   └── 08_paper_analysis/         # Paper Analysis 时延分析
│
└── README.md
```

---

## 五种输入方式 (`datagen/`)

同一 Transformer Encoder (Q/K/V → Softmax → O → LN → FFN_up → GELU → FFN_down → LN)，
五种前端写法，全部使用 bfp8 精度。

| # | 目录 | 输入方式 | 面向人群 |
|---|------|---------|---------|
| 0 | `00_datagen_manual` | DataGenerator 手动 API | 开发者/精细控制 |
| 1 | `01_datagen_autogen` | 算子自动生成 API | 开发者/快速验证 |
| 2 | `02_model_dsl` | Model DSL (类 PyTorch) | 算法团队 |
| 3 | `03_torch_golden` | PyTorch 劫持 (Golden Mode) | 算法团队/现有代码 |
| 4 | `04_xlsx_config` | Excel/XLSX 配置 | 硬件测试/非编程 |

```bash
python demos/datagen/00_datagen_manual/run.py
python demos/datagen/01_datagen_autogen/run.py
python demos/datagen/02_model_dsl/run.py
python demos/datagen/03_torch_golden/run.py
python demos/datagen/04_xlsx_config/run.py
```

---

## 四种比数 (`compare/`)

| Track | 名称 | 说明 | 典型场景 |
|-------|------|------|---------|
| 1 | golden_pure | 纯 fp32 计算的 golden（基准） | 所有比对的参考基线 |
| 2 | golden_local | 本地格式 (fp16/int8) 原生数据 golden | 浮点降精度影响评估 |
| 3 | golden_hw | 硬件格式 (bfp/gfp) 量化→反量化 golden | 硬件量化误差评估 |
| 4 | golden_qa | 量化感知受控随机 golden | 动态范围可控的比对 |

### compare/06_encoder_bfp8 — Encoder 全局 bfp8 四种比数

三种前端模式 (DSL / Torch / XLSX) 对同一 Encoder 生成 golden，全局 bfp8:
- 三种模式输出对比 + 跨模式一致性验证
- 10 个算子逐层四种比数: pure vs local / pure vs hw / pure vs qa
- DUT 模拟 (bfp8 + noise) vs golden

```bash
python demos/compare/06_encoder_bfp8/run.py
```

### compare/07_qa_encoder_bfp8 — 量化感知随机 bfp8 比数

对比实验: 正态随机 vs 量化感知随机 (QA-aware)，评估 QA 模式对 bfp8 比数的影响:
- A 组: 标准正态分布 N(0,1)
- B 组: QA 受控范围 (center=1.0, amplitude=0.5)
- 数据特征分析 + Golden 漂移分析 + 动态范围对比

```bash
python demos/compare/07_qa_encoder_bfp8/run.py
```

---

## 优化器 (`optimizer/`)

时延分析、融合评估、ML 校准。

| # | 目录 | 功能 |
|---|------|------|
| 1 | `01_basic` | PyTorch 劫持 → Benchmark 提取 → 时延评估 |
| 2 | `02_calibration` | ML 校准: 生成测试用例 → 导入实测 → 校准超参数 |
| 3 | `03_fusion_rules` | 融合规则: 全局配置、多算子模式、超参数 |
| 4 | `04_echarts` | ECharts 可视化: 柱状图、饼图、Roofline、热力图 |
| 5 | `05_comparison` | 理论 vs 工程化方法精度对比 |
| 6 | `06_bridge` | PyTorch → Benchmark 自动桥接 |

```bash
python demos/optimizer/01_basic/run.py
python demos/optimizer/02_calibration/run.py
python demos/optimizer/03_fusion_rules/run.py
python demos/optimizer/04_echarts/run.py
python demos/optimizer/05_comparison/run.py
python demos/optimizer/06_bridge/run.py
```

---

## 入门 Demo

### 01_basic_ops — 基础算子 (PyTorch 劫持)

```bash
python demos/01_basic_ops/run.py
```

### 02_mini_transformer — 完整比对流程 (推荐入门)

```bash
python demos/02_mini_transformer/run.py
```

### 03_transformer — 完整 Transformer

```bash
python demos/03_transformer/run.py
```

---

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
