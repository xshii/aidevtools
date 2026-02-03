# Block Floating Point (BFP) 格式说明

## 概述

Block Floating Point (BFP) 是一种数据量化格式，将数据分成固定大小的块，每个块内的元素共享一个指数，每个元素只存储尾数。相比标准浮点格式，BFP 在保持一定精度的同时显著降低存储和计算开销。

## 数据结构图

```
┌─────────────────────────────────────────────────────────────────┐
│                        FP32 原始数据                             │
│  [3.5, 1.2, -2.8, 0.5, 0.1, 0.3, -0.2, 0.4, ...]               │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼ 分块 (block_size=4)
┌───────────────────┐   ┌───────────────────┐
│      Block 0      │   │      Block 1      │   ...
│ [3.5, 1.2, -2.8,  │   │ [0.1, 0.3, -0.2,  │
│        0.5]       │   │        0.4]       │
└───────────────────┘   └───────────────────┘
         │                       │
         ▼                       ▼
┌───────────────────┐   ┌───────────────────┐
│ max_abs = 3.5     │   │ max_abs = 0.4     │
│ shared_exp = 2    │   │ shared_exp = 0    │
│ (2^2=4 > 3.5)     │   │ (2^0=1 > 0.4)     │
└───────────────────┘   └───────────────────┘
         │                       │
         ▼                       ▼
┌───────────────────┐   ┌───────────────────┐
│    Mantissas      │   │    Mantissas      │
│ (8-bit, 范围±127) │   │ (8-bit, 范围±127) │
│ [111, 38, -89, 16]│   │ [12, 38, -25, 51] │
└───────────────────┘   └───────────────────┘
                              │
                              ▼ 打包存储
┌─────────────────────────────────────────────────────────────────┐
│              BFP 打包格式 (int8 数组)                            │
│ [exp0, exp1, ..., m0, m1, m2, m3, m4, m5, m6, m7, ...]          │
│  └─ 共享指数 ─┘   └────────────── 尾数 ──────────────┘           │
└─────────────────────────────────────────────────────────────────┘
```

## 量化示例

以 `block_size=4, mantissa_bits=8` 为例：

### 输入数据

```python
data = [3.5, 1.2, -2.8, 0.5]  # Block 0
```

### Step 1: 计算共享指数

```
max_abs = max(|3.5|, |1.2|, |-2.8|, |0.5|) = 3.5
shared_exp = floor(log2(3.5)) + 1 = floor(1.807) + 1 = 2
```

### Step 2: 计算尾数

```
scale = 2^(mantissa_bits - 1 - shared_exp) = 2^(8 - 1 - 2) = 2^5 = 32

mantissa[0] = round(3.5 × 32) = round(112) = 112
mantissa[1] = round(1.2 × 32) = round(38.4) = 38
mantissa[2] = round(-2.8 × 32) = round(-89.6) = -90
mantissa[3] = round(0.5 × 32) = round(16) = 16
```

### Step 3: 反量化验证

```
scale = 2^(shared_exp - (mantissa_bits - 1)) = 2^(2 - 7) = 2^(-5) = 0.03125

restored[0] = 112 × 0.03125 = 3.5      ✓ 精确
restored[1] = 38 × 0.03125 = 1.1875    ≈ 1.2 (误差 1%)
restored[2] = -90 × 0.03125 = -2.8125  ≈ -2.8 (误差 0.4%)
restored[3] = 16 × 0.03125 = 0.5       ✓ 精确
```

### 精度损失分析

块内最大值 (3.5) 精度最高，较小值 (1.2) 有轻微损失。这是 BFP 的核心特性：**块内共享指数，大值精度高，小值精度低**。

## 预设格式

| 格式名 | block_size | mantissa_bits | 适用场景 |
|--------|------------|---------------|----------|
| `bfp16` | 16 | 8 | 通用量化，精度较高 |
| `bfp8` | 32 | 4 | 激进量化，适合推理 |

## 参数说明

### block_size

- **含义**：每个块包含的元素数量
- **默认值**：bfp16=16, bfp8=32
- **取值范围**：通常为 4, 8, 16, 32, 64
- **影响**：
  - 越大 → 压缩率越高，但块内动态范围大时精度损失增加
  - 越小 → 精度越高，但指数存储开销增加

### mantissa_bits

- **含义**：每个元素尾数的位宽
- **默认值**：bfp16=8, bfp8=4
- **取值范围**：通常为 4, 8, 12, 16
- **影响**：
  - 越大 → 块内精度越高
  - 越小 → 存储越紧凑，精度损失越大

## 使用方法

### 基本使用

```python
from aidevtools.formats.quantize import quantize
import numpy as np

data = np.array([3.5, 1.2, -2.8, 0.5], dtype=np.float32)

# 使用默认参数
packed, meta = quantize(data, "bfp16")
print(meta)
# {'format': 'bfp', 'block_size': 16, 'mantissa_bits': 8, ...}
```

### 自定义参数

```python
# 覆盖 block_size
packed, meta = quantize(data, "bfp16", block_size=4)

# bfp8 使用更小的块
packed, meta = quantize(data, "bfp8", block_size=16)
```

### 直接调用 Golden API

```python
from aidevtools.formats.custom.bfp.golden import fp32_to_bfp, bfp_to_fp32

# 量化
mantissas, shared_exps, meta = fp32_to_bfp(
    data,
    block_size=4,
    mantissa_bits=8
)

print(f"共享指数: {shared_exps}")  # [2]
print(f"尾数: {mantissas}")        # [112, 38, -90, 16]

# 反量化
restored = bfp_to_fp32(
    mantissas,
    shared_exps,
    block_size=4,
    mantissa_bits=8,
    original_shape=data.shape
)
print(f"还原: {restored}")  # [3.5, 1.1875, -2.8125, 0.5]
```

## C++ Golden 实现

BFP 提供 C++ Golden 实现用于精度验证：

```python
from aidevtools.formats.custom.bfp import register_bfp_golden

# 注册 C++ golden 实现
register_bfp_golden()

# 之后 quantize 调用 bfp16_golden/bfp8_golden 会使用 C++ 版本
packed, meta = quantize(data, "bfp16_golden")
```

### 检查 C++ 是否可用

```python
from aidevtools.formats.custom.bfp import is_cpp_available

if is_cpp_available():
    print("C++ BFP Golden 可用")
else:
    print("需要编译: cd src/aidevtools/formats/custom/bfp/cpp && bash build.sh")
```

## 精度特性

BFP 的量化误差取决于两个因素：

1. **块内动态范围**：如果块内数值差异很大，小数值会损失更多精度
2. **尾数位宽**：8位尾数约有 1/128 ≈ 0.8% 的基础量化误差

### 精度优化建议

- 数据分布均匀时使用较大 block_size
- 数据动态范围大时使用较小 block_size
- 对精度敏感的场景使用 bfp16 (8位尾数)
- 推理场景可使用 bfp8 (4位尾数)

## 存储格式

打包后的数据布局：

```
┌─────────────────────────────────────────────────────────────┐
│ [exp_0, exp_1, ..., exp_n | m_0, m_1, m_2, ..., m_k]       │
│  └── num_blocks 个 ──────┘ └──── num_elements 个 ────┘      │
└─────────────────────────────────────────────────────────────┘
```

其中：
- `num_blocks = ceil(num_elements / block_size)`
- 每个 shared_exp 和 mantissa 都是 int8

## 参考资料

- [AMD Quark BFP16](https://quark.docs.amd.com/latest/onnx/tutorial_bfp16_quantization.html)
- [Static BFP CNN](https://github.com/os-hxfan/Static_BFP_CNN)
