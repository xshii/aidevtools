# AIDevTools 完整架构设计

## 一、项目总览

```
aidevtools/
├── analysis/       # 性能分析 (Roofline, Pass链)
├── compare/        # 4状态比对系统
├── frontend/       # 前端 (数据生成, 编译)
├── ops/            # 算子 (TracedTensor, 计算图)
├── golden/         # PyTorch 劫持
├── optimizer/      # 融合优化 (本模块)
├── formats/        # 数据格式 (GFloat, BFP)
├── toolchain/      # 编译器工具链
├── core/           # 全局配置
├── trace/          # 执行轨迹
└── xlsx/           # Excel 转换
```

## 二、完整数据流

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        3 种前端写作方式                                  │
├───────────────────┬─────────────────────┬───────────────────────────────┤
│  1. Python DSL    │  2. Benchmark API   │  3. PyTorch 劫持              │
│  (frontend)       │  (optimizer)        │  (golden + ops)               │
│                   │                     │                               │
│  gen = DataGen()  │  Benchmark("ffn")   │  import aidevtools.golden     │
│  x = gen.gen_input│  .add_op("mm",...)  │  y = F.linear(x, w)           │
│  compile_to_dut() │  .add_op("gelu",..) │  # 自动劫持+比对              │
└─────────┬─────────┴──────────┬──────────┴───────────────┬───────────────┘
          │                    │                          │
          ▼                    ▼                          ▼
┌─────────────────┐  ┌─────────────────────┐  ┌───────────────────────────┐
│ Tensor/TensorMeta│  │ Benchmark/OpSpec    │  │ TracedTensor              │
│ (frontend.types) │  │ (optimizer.benchmark)│  │ (ops.traced_tensor)       │
└────────┬────────┘  └──────────┬──────────┘  └─────────────┬─────────────┘
         │                      │                           │
         └──────────────────────┼───────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         OpProfile (中间表示)                             │
│                    (analysis.profile.OpProfile)                         │
└───────────────────────────────┬─────────────────────────────────────────┘
                                │
          ┌─────────────────────┼─────────────────────┐
          ▼                     ▼                     ▼
┌──────────────────┐  ┌──────────────────┐  ┌──────────────────────────────┐
│  analysis 模块   │  │  optimizer 模块  │  │     compare 模块             │
│  ──────────────  │  │  ──────────────  │  │     ──────────────           │
│  ChipSpec        │  │  FusionRules     │  │     4状态判定:               │
│  PaperAnalyzer   │  │  HyperCalibrator │  │     ┌─────────┬────────────┐ │
│  Pass 链:        │  │  CostModel       │  │     │DUT\Gold │ PASS  FAIL │ │
│  - Roofline      │  │  TilingStrategy  │  │     ├─────────┼────────────┤ │
│  - MemEfficiency │  │                  │  │     │ PASS    │ PASS  SUSPECT│ │
│  - Overhead      │  │  理论 vs 工程化  │  │     │ FAIL    │ ISSUE BOTH  │ │
│  - ...           │  │  comparison      │  │     └─────────┴────────────┘ │
└────────┬─────────┘  └────────┬─────────┘  └──────────────┬───────────────┘
         │                     │                           │
         ▼                     ▼                           ▼
   LatencyResult         EvalResult              CompareResult
   (理论时延)            (优化时延)              (比对状态)
         │                     │                           │
         └─────────────────────┼───────────────────────────┘
                               ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                            输出层                                        │
├─────────────────┬─────────────────────┬─────────────────────────────────┤
│  export_xlsx    │  ECharts 可视化     │  CompareReport                  │
│  export_csv     │  Roofline 图        │  JSON/Text 报告                 │
│  export_json    │  HTML 图表          │  状态汇总                       │
└─────────────────┴─────────────────────┴─────────────────────────────────┘
```

## 三、Optimizer 模块架构

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           前端 (3种写作方式)                             │
├─────────────────────┬─────────────────────┬─────────────────────────────┤
│   1. Python DSL     │   2. Benchmark API  │   3. 实测数据导入            │
│   (frontend 模块)   │   (链式构建)        │   (CSV/Dict)                │
├─────────────────────┴─────────────────────┴─────────────────────────────┤
│                                                                         │
│  from frontend import   Benchmark("ffn")      archive.import_results(   │
│    Tensor, compile      .add_op("mm1",...)    [("bm1", 125.5), ...],    │
│                         .add_op("gelu",...)    suite)                   │
│                                                                         │
└─────────────────────────────────┬───────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         Benchmark / OpSpec                              │
│                    (统一的算子描述中间表示)                              │
└─────────────────────────────────┬───────────────────────────────────────┘
                                  │
                    ┌─────────────┴─────────────┐
                    ▼                           ▼
┌───────────────────────────────┐ ┌───────────────────────────────────────┐
│      理论分析 (analysis)      │ │        工程化方法 (optimizer)          │
├───────────────────────────────┤ ├───────────────────────────────────────┤
│                               │ │                                       │
│  ChipSpec (芯片规格)          │ │  FusionHyperParams (ML超参数)         │
│  OpProfile (算子画像)         │ │  MeasurementArchive (实测数据)        │
│  RooflinePass (Roofline分析)  │ │  HyperCalibrator (超参数校准)         │
│  MemoryEfficiencyPass         │ │  ParameterizedCostModel               │
│  OverheadPass                 │ │                                       │
│                               │ │                                       │
│  基于芯片规格推导             │ │  基于实测数据学习                     │
│  (优化前默认参数)             │ │  (ML校准后参数)                       │
│                               │ │                                       │
└───────────────────┬───────────┘ └───────────────────┬───────────────────┘
                    │                                 │
                    ▼                                 ▼
┌───────────────────────────────┐ ┌───────────────────────────────────────┐
│   理论预测时延 (latency_us)   │ │   工程化预测时延 (latency_us)          │
└───────────────────┬───────────┘ └───────────────────┬───────────────────┘
                    │                                 │
                    └─────────────┬───────────────────┘
                                  ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                     comparison (对比模块)                               │
├─────────────────────────────────────────────────────────────────────────┤
│  MethodComparator.from_calibration()                                    │
│  - 对比理论分析 vs 工程化方法                                           │
│  - 输出 MAE/MAPE/RMSE/R² 等指标                                         │
│  - 生成校准后的时延预测                                                 │
└─────────────────────────────────────────────────────────────────────────┘
```

## 三种前端写作方式

### 1. Python DSL (frontend 模块)
```python
from aidevtools.frontend import Tensor, DataGenerator, compile_to_dut

# 数据生成
gen = DataGenerator(seed=42)
x = gen.gen_input(shape=(2, 64), dtype="bfp16")
w = gen.gen_weight(shape=(64, 64), dtype="bfp16")

# 编译到 DUT
result = compile_to_dut(source="model.py", output="build/model.bin")
```

### 2. Benchmark API (链式构建)
```python
from aidevtools.optimizer import Benchmark, BenchmarkSuite

# 链式构建
bm = (
    Benchmark("my_ffn")
    .add_op("mm1", "matmul", M=512, N=3072, K=768)
    .add_op("gelu", "gelu", M=512, N=3072)
    .add_op("mm2", "matmul", M=512, N=768, K=3072)
)

# 或使用预定义 Suite
suite = BenchmarkSuite()
bm = suite.get("bert_ffn_512")
```

### 3. 实测数据导入
```python
from aidevtools.optimizer import MeasurementArchive, BenchmarkSuite

archive = MeasurementArchive()
suite = BenchmarkSuite()

# 只需 benchmark 名称 + 时延
results = [
    ("bert_ffn_512", 125.5),
    ("gpt_attention_1024", 380.0),
]
archive.import_results(results, suite)
```

## 后端时延评估框架

### 理论分析方法 (analysis 模块)
- 基于芯片规格 (ChipSpec) 和算子画像 (OpProfile)
- 使用 Roofline 模型、访存效率模型
- 参数是静态的，基于硬件规格推导
- 优点：无需实测数据
- 缺点：可能与实际有偏差

### 工程化方法 (optimizer 模块)
- 基于实测数据 (MeasurementArchive)
- 使用 ML 校准 (HyperCalibrator)
- 参数是动态的，从数据中学习
- 优点：更贴近实际
- 缺点：需要实测数据

### 对比流程
```python
from aidevtools.optimizer import calibrate_and_compare

# 一键完成: 校准 + 对比 + 输出校准后时延
result = calibrate_and_compare(archive)

# 查看对比结果
print(result.summary())
# 理论分析 MAPE: 12.5%
# 工程化 MAPE: 3.2%
# 结论: 工程化方法更优

# 获取校准后的时延预测
for bm, latency in result.calibrated_predictions.items():
    print(f"{bm}: {latency:.2f} us")
```

## 数据流

```
前端输入                    中间表示                    后端输出
────────                    ────────                    ────────

Python DSL ──┐
             │
Benchmark ───┼──▶ OpSpec ──▶ OpProfile ──┬──▶ 理论时延
API          │              FeatureVector │
             │                            ├──▶ 工程化时延
实测数据 ────┘              LabelVector ──┤
                                          └──▶ 对比报告
```

## 四、模块依赖关系

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          aidevtools (主入口)                            │
└────────────────────────────────┬────────────────────────────────────────┘
                                 │
    ┌────────────┬───────────────┼───────────────┬────────────┐
    ▼            ▼               ▼               ▼            ▼
┌────────┐ ┌──────────┐ ┌────────────┐ ┌─────────────┐ ┌──────────┐
│frontend│ │  golden  │ │   compare  │ │   analysis  │ │ optimizer│
└───┬────┘ └────┬─────┘ └─────┬──────┘ └──────┬──────┘ └────┬─────┘
    │           │             │               │              │
    │           ▼             │               │              │
    │      ┌─────────┐        │               │              │
    │      │   ops   │◄───────┤               │              │
    │      └────┬────┘        │               │              │
    │           │             │               │              │
    │           ▼             │               │              │
    │      ┌─────────┐        │               │              │
    └─────►│ formats │◄───────┴───────────────┴──────────────┘
           │(GFloat, │
           │  BFP)   │
           └────┬────┘
                │
                ▼
           ┌─────────┐
           │  core   │ (config, log, utils)
           └─────────┘
```

## 五、各模块核心职责

| 模块 | 职责 | 关键类/函数 |
|------|------|-------------|
| **analysis** | 性能分析、Roofline、Pass链 | PaperAnalyzer, ChipSpec, RooflinePass |
| **frontend** | 数据生成、编译接口 | DataGenerator, Tensor, compile_to_dut |
| **compare** | 4状态比对系统 | CompareEngine, CompareStatus |
| **ops** | 算子实现、计算图追踪 | TracedTensor, cpu_golden |
| **golden** | PyTorch劫持 | TorchGoldenBackend, golden_mode |
| **optimizer** | 融合评估、ML校准 | FusionEvaluator, HyperCalibrator |
| **formats** | 数据格式、量化 | GFloat, BFP, quantize |
| **toolchain** | 编译器工具链 | ToolchainManager, get_compiler |
| **core** | 全局配置 | GlobalConfig, logger |

## 六、关键工作流

### 工作流 1: PyTorch Golden 模式
```python
import aidevtools.golden as golden

# 配置
golden.set_mode("python")      # cpp/python/none
golden.set_compare("fuzzy")    # exact/fuzzy/none
golden.set_quantize("bfp16")   # gfp16/bfp16/...

# 执行 (自动劫持)
y = torch.nn.functional.linear(x, w)

# 查看结果
golden.report()
```

### 工作流 2: 性能分析
```python
from aidevtools.analysis import PaperAnalyzer, load_chip_spec

analyzer = PaperAnalyzer(chip="npu_910")
analyzer.add_profile(profile)
result = analyzer.analyze()
analyzer.print_summary()
```

### 工作流 3: 融合优化评估
```python
from aidevtools.optimizer import (
    Benchmark, FusionEvaluator, calibrate_and_compare
)

# 定义 benchmark
bm = Benchmark("ffn").add_op("mm", "matmul", M=512, N=768, K=768)

# 评估
evaluator = FusionEvaluator()
result = evaluator.evaluate(bm)

# 校准并对比
comparison = calibrate_and_compare(archive)
print(comparison.summary())
```

### 工作流 4: 4状态比对
```python
from aidevtools.compare import compare_full, CompareStatus

result = compare_full(dut_output, golden_pure, golden_quantized)

if result.status == CompareStatus.PASS:
    print("✓ 比对通过")
elif result.status == CompareStatus.DUT_ISSUE:
    print("✗ DUT 问题")
elif result.status == CompareStatus.GOLDEN_SUSPECT:
    print("? Golden 可疑")
else:
    print("? 双方可疑")
```

## 七、optimizer 模块内部结构

```
optimizer/
├── benchmark.py           # Benchmark 链式构建
├── fusion_rules.py        # 全局融合规则 (Singleton)
├── cost_model.py          # 成本模型 (复用 analysis)
├── measurement_archive.py # 实测数据归档 (X, Y)
├── hyper_calibrator.py    # 超参数 ML 校准
├── calibration.py         # 传统校准方法
├── comparison.py          # 理论 vs 工程化对比
├── evaluator.py           # Facade 评估器
├── memory_plan.py         # 内存规划
├── strategy/              # Tiling 策略
│   ├── base.py            # 策略基类
│   ├── baseline.py        # 基准策略
│   ├── efficiency_aware.py# 效率导向
│   └── fuse_speedup.py    # 加速比导向
├── views/                 # 可视化
│   ├── base.py            # 视图基类
│   ├── roofline.py        # Roofline 图
│   ├── echarts.py         # ECharts 转换
│   └── ...
├── demos/                 # 演示脚本
│   ├── 01_basic.py
│   ├── 02_calibration.py
│   ├── 03_fusion_rules.py
│   ├── 04_echarts.py
│   └── 05_comparison.py
└── ARCHITECTURE.md        # 本文档
```
