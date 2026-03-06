# 数据生成模块设计说明书

> 版本: 3.0
> 日期: 2026-03
> 模块: `aidevtools.datagen` + `aidevtools.ops.datagen` + `aidevtools.core.random`

---

## 1. 概述

### 1.1 目的

数据生成模块提供 DUT 测试所需的输入数据、权重、Golden 输出的自动生成能力，覆盖五种使用方式：

1. **手动 API** — 精细控制每个张量
2. **算子自动生成** — 基于 `@register_op` 元信息自动生成
3. **Model DSL** — 类 PyTorch 语法构建模型
4. **PyTorch 劫持** — 标准 PyTorch 代码自动记录 Golden
5. **Excel/XLSX 配置** — 非编程人员通过 Excel 定义算子

### 1.2 设计原则

1. **统一入口**: `DataGenerator` 和 `Model` 作为主 API，内部复用 `RandomGenerator`
2. **四种比数**: 一次调用生成 Pure / Local / HW / QA 四种 Golden
3. **L2 内存管理**: 所有生成数据自动分配 L2 地址 (256 字节对齐)
4. **精度可配**: 通过 `PrecisionConfig` 控制算子级精度 (input/weight/output dtype)

---

## 2. 模块架构

### 2.1 文件结构

```
aidevtools/
├── datagen.py              # 统一数据生成器 (DataGenerator, Model, FourTrackGolden)
├── core/
│   └── random.py           # 公共随机数生成器 (RandomGenerator)
└── ops/
    └── datagen.py           # 算子数据生成器 (OpDataGenerator, L2MemoryLayout)
```

### 2.2 依赖关系

```
datagen.py ──→ core/random.py (随机数生成)
           ──→ ops/registry   (算子元信息)
           ──→ formats/quantize (量化/反量化)
           ──→ frontend/types  (PrecisionConfig)

ops/datagen.py ──→ core/random.py
               ──→ ops/registry
```

### 2.3 代码量

| 文件 | 行数 | 职责 |
|------|------|------|
| `datagen.py` | 1077 | DataGenerator + Model DSL + FourTrackGolden |
| `ops/datagen.py` | 552 | OpDataGenerator + L2MemoryLayout |
| `core/random.py` | 574 | RandomGenerator + 策略解析 |
| **合计** | **2203** | |

---

## 3. 核心数据结构

### 3.1 GeneratedTensor (datagen.py)

```python
@dataclass
class GeneratedTensor:
    name: str              # 张量名
    array: np.ndarray      # 数据
    l2_addr: int = 0       # L2 绝对地址
    l2_size: int = 0       # L2 占用字节
    qtype: str = "fp32"    # 量化类型
    role: str = "input"    # input / weight / output
```

### 3.2 FourTrackGolden (datagen.py)

```python
@dataclass
class FourTrackGolden:
    golden_pure: np.ndarray                    # Track 1: 纯 fp32
    golden_local: Optional[np.ndarray] = None  # Track 2: 本地格式 (fp16/int8)
    golden_hw: Optional[np.ndarray] = None     # Track 3: 硬件格式 (bfpp8/bfpp4)
    golden_qa: Optional[np.ndarray] = None     # Track 4: 量化感知
    data_pure: Optional[Dict] = None           # 各 Track 使用的输入数据
    data_local: Optional[Dict] = None
    data_hw: Optional[Dict] = None
    data_qa: Optional[Dict] = None

    @property
    def all_goldens(self) -> Dict[str, np.ndarray]:
        """返回所有非 None 的 golden"""
```

### 3.3 TensorInfo / L2MemoryLayout (ops/datagen.py)

```python
@dataclass
class TensorInfo:
    name: str
    data: np.ndarray
    shape: Tuple[int, ...]
    dtype: str = "fp32"
    qtype: str = "bfpp16"
    l2_offset: int = 0     # L2 偏移
    l2_addr: int = 0       # L2 绝对地址
    size: int = 0           # 字节数
    role: str = "input"
    op_name: str = ""

@dataclass
class L2MemoryLayout:
    tensors: List[TensorInfo]
    alignment: int = 256     # 256 字节对齐
    l2_base: int = 0x100000
    current_offset: int = 0
```

### 3.4 RandomGenerator (core/random.py)

```python
class RandomGenerator:
    def __init__(self, seed=None): ...

    # 基础生成
    def normal(shape) -> np.ndarray
    def uniform(shape, low, high) -> np.ndarray
    def zeros(shape) -> np.ndarray
    def ones(shape) -> np.ndarray
    def xavier(shape) -> np.ndarray
    def kaiming(shape) -> np.ndarray

    # 量化感知
    def qa_uniform(shape, center, amplitude) -> np.ndarray
    def set_qa_config(enabled, center, amplitude)

    # 策略解析 (供 datagen 使用)
    def generate_from_strategy(strategy: str, context: dict) -> (ndarray, shape)
```

---

## 4. API 详细设计

### 4.1 DataGenerator — 统一数据生成器

```python
from aidevtools import DataGenerator

gen = DataGenerator(
    seed=42,
    l2_base=0x100000,     # L2 基地址
    alignment=256,         # 对齐字节数
    qtype="bfpp16",        # 默认量化类型
    precision=None,        # PrecisionConfig (可选)
)
```

#### 手动生成

```python
x = gen.randn((512, 768), name="input", qtype="bfpp8")
w = gen.xavier((3072, 768), name="weight", qtype="bfpp4")
b = gen.zeros((3072,), name="bias")
```

#### 自动生成 (基于 @register_op)

```python
data = gen.generate("linear", input_shape=(512, 768), out_features=3072)
# data = {"input": GeneratedTensor, "weight": GeneratedTensor, "bias": GeneratedTensor}
```

#### 生成 + Golden

```python
data, golden = gen.generate_with_golden("linear", input_shape=(512, 768), out_features=3072)
```

#### 四种比数

```python
from aidevtools.frontend.types import PrecisionConfig

pc = PrecisionConfig(
    input_dtype="fp16",
    weight_dtype="bfpp4",
    compute_dtype="fp32",
    output_dtype="bfpp8",
    qa_aware=True,
)

tracks = gen.generate_four_track("linear", input_shape=(512, 768),
                                  precision=pc, out_features=3072)
# tracks.golden_pure   — fp32 计算
# tracks.golden_local  — fp16 原生数据计算
# tracks.golden_hw     — bfpp4 量化→反量化后计算
# tracks.golden_qa     — 量化感知随机权重计算
```

#### 导出

```python
# DUT 格式 (量化后二进制)
gen.export("./golden/", bm="encoder")

# C 头文件 (L2 地址宏定义)
gen.export_header("./golden/memory_layout.h")

# 内存摘要
print(gen.memory_summary())
```

### 4.2 Model DSL — 类 PyTorch 语法

```python
from aidevtools import Model
from aidevtools.frontend.types import PrecisionConfig

pc = PrecisionConfig(input_dtype="fp16", weight_dtype="bfpp4",
                     compute_dtype="fp32", qa_aware=True)

with Model(seed=42, precision=pc) as m:
    x = m.input((2, 16, 64))

    # Self-Attention
    q = m.linear(x, out_features=64)
    k = m.linear(x, out_features=64)
    v = m.linear(x, out_features=64)
    attn = m.softmax(q)
    out = m.linear(attn, out_features=64)
    ln1 = m.layernorm(out)

    # FFN
    up = m.linear(ln1, out_features=256)
    act = m.gelu(up)
    down = m.linear(act, out_features=64)
    output = m.layernorm(down)

# 访问结果
print(m.final_output.shape)     # 最终 golden
print(len(m.tensors))           # 所有生成的张量
print(len(m.outputs))           # 各层输出

m.export("./golden/", bm="encoder")
```

**支持的算子**: `linear`, `gelu`, `relu`, `silu`, `softmax`, `layernorm`, `rmsnorm`, `matmul`, `add`, `mul`

### 4.3 OpDataGenerator — 算子级生成 (底层)

```python
from aidevtools.ops.datagen import OpDataGenerator

gen = OpDataGenerator(seed=42, l2_base=0x100000, qtype="bfpp8")
data = gen.generate("linear", input_shape=(512, 768), out_features=3072)

# L2 内存布局
layout = gen.memory_layout()
print(layout.summary())

# 导出
gen.export_dut("./golden/")
gen.export_header("./golden/layout.h")

# 创建 MemoryPlan (用于 DMA 生成)
plan = gen.create_memory_plan(l1_size=256*1024, l2_size=2*1024*1024)
```

---

## 5. 四种比数 (Four Track Golden)

### 5.1 设计

```
Track 1: golden_pure   — 纯 fp32 计算的 golden（基准参考）
Track 2: golden_local  — 本地格式 (fp16/int8) 原生数据计算
Track 3: golden_hw     — 硬件格式 (bfpp/gfp) 量化→反量化后计算
Track 4: golden_qa     — 量化感知随机权重计算
```

### 5.2 生成流程

```
generate_four_track(op_name, input_shape, precision):
    1. Track 1: 纯 fp32 数据 → cpu_golden → golden_pure
    2. Track 2: fp32 → cast 到本地格式 (fp16/int8) → cpu_golden → golden_local
    3. Track 3: fp32 → simulate_quantize(bfpp8/bfpp4) → cpu_golden → golden_hw
    4. Track 4: qa_aware 随机数 → cpu_golden → golden_qa
```

### 5.3 本地格式数据生成

```python
def _to_native_local_dtype(data_fp32, local_dtype, rng=None):
    """
    fp16/bf16: 直接 cast (原生精度)
    int8/int16: 直接生成随机整数 (不经过 fp32 中转)
    fp32: 原样返回
    """
```

---

## 6. L2 内存管理

### 6.1 对齐策略

所有张量在 L2 中按 256 字节对齐:

```
L2 Base: 0x100000
Tensor 0: 0x100000 (1024 bytes)
Tensor 1: 0x100500 (aligned to 0x100500, not 0x100400)
Tensor 2: ...
```

### 6.2 导出格式

**DUT 二进制**: `{bm}_{timestamp}_{name}_{shape}.{qtype}.bin`

```
encoder_20260306_143025_linear_0_weight_64x64.bfpp4.bin
```

**C 头文件**:
```c
#define DATA_L2_BASE 0x00100000
#define DATA_LINEAR_0_WEIGHT_ADDR 0x00100000
#define DATA_LINEAR_0_WEIGHT_SIZE 4096
#define DATA_LINEAR_0_WEIGHT_QTYPE QTYPE_BFP4
```

---

## 7. 量化感知 (QA) 随机数

### 7.1 原理

标准随机数动态范围大，量化损失高。QA 模式生成受控动态范围的数据:

```python
# 标准: randn() → 动态范围 [-3, 3]，量化损失大
# QA:   center ± amplitude → 动态范围 [0.5, 1.5]，量化损失小
```

### 7.2 配置

```python
PrecisionConfig(
    qa_aware=True,        # 启用 QA
    qa_center=1.0,        # 中心值
    qa_amplitude=0.5,     # 振幅
)
```

### 7.3 效果

| 场景 | QSNR (bfpp8) | 说明 |
|------|------------|------|
| 标准随机数 | ~17 dB | 动态范围大，量化损失高 |
| QA 随机数 | ~32 dB | 受控范围，量化损失低 |

---

## 8. Demo 索引

| Demo | 方式 | 精度 | 说明 |
|------|------|------|------|
| `datagen_00_manual` | DataGenerator 手动 API | bfpp8/bfpp4 | 精细控制每个张量 |
| `datagen_01_autogen` | generate_four_track() | fp16/bfpp4 | 四种比数自动生成 |
| `datagen_02_dsl` | Model DSL | fp16/bfpp4 | 类 PyTorch 语法 |
| `datagen_03_torch` | PyTorch 劫持 | bfpp8/bfpp4 | 标准 PyTorch 代码 |
| `datagen_04_xlsx` | Excel 配置 | bfpp8/bfpp4 | 非编程人员使用 |

---

## 9. 与其他模块的关系

```
PrecisionConfig (frontend/types)
    ↓
DataGenerator / Model DSL (datagen.py)
    ├── RandomGenerator (core/random.py)    — 随机数生成
    ├── ops/registry                         — 算子元信息 (@register_op)
    ├── ops/cpu_golden                       — Golden 计算
    ├── formats/quantize                     — 量化/反量化
    └── OpDataGenerator (ops/datagen.py)     — L2 内存管理
            └── MemoryPlan (optimizer)       — DMA 规划 (延迟导入)
```

---

## 10. 参考

- [Demo 示例](../../demos/datagen/)
- [比对模块设计](./compare_module_design.md)
- [源代码](../../aidevtools/datagen.py)
