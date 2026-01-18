# GFloat 自定义浮点格式说明

## 概述

GFloat 是一种简单的浮点数截断格式，通过截取 FP32 的高位来实现快速量化。相比 BFP 的块共享指数方案，GFloat 每个元素独立存储，实现更简单但压缩率较低。

## 数据结构图

### FP32 原始格式 (32 bits)

```
┌─────┬──────────────────┬───────────────────────────────────┐
│ S   │    Exponent      │            Mantissa               │
│ 1b  │      8 bits      │             23 bits               │
├─────┼──────────────────┼───────────────────────────────────┤
│  31 │   30  ...  23    │      22  ...  ...  ...  0         │
└─────┴──────────────────┴───────────────────────────────────┘
```

### GFloat16 格式 (16 bits) - 截取高 16 位

```
┌─────┬──────────────────┬─────────────────┐
│ S   │    Exponent      │    Mantissa     │
│ 1b  │      8 bits      │     7 bits      │
├─────┼──────────────────┼─────────────────┤
│  15 │   14  ...  7     │   6  ...  0     │
└─────┴──────────────────┴─────────────────┘

FP32:    [S|EEEEEEEE|MMMMMMMMMMMMMMMMMMMMMMM]
              ↓ 截取高16位
GFloat16: [S|EEEEEEEE|MMMMMMM]
```

### GFloat8 格式 (8 bits) - 截取高 8 位

```
┌─────┬──────────┬───────┐
│ S   │ Exponent │Mantis │
│ 1b  │  4 bits  │ 3 bits│
├─────┼──────────┼───────┤
│  7  │  6 ... 3 │2 ... 0│
└─────┴──────────┴───────┘

FP32:    [S|EEEEEEEE|MMMMMMMMMMMMMMMMMMMMMMM]
              ↓ 截取高8位
GFloat8:  [S|EEEE|MMM]
```

## 量化示例

### 输入数据

```python
value = 3.5  # FP32
```

### FP32 二进制表示

```
3.5 = 1.75 × 2^1

符号位 S = 0 (正数)
指数 E = 1 + 127 = 128 = 0b10000000
尾数 M = 0.75 = 0b11000000000000000000000

FP32 bits: 0 10000000 11000000000000000000000
         = 0x40600000
```

### GFloat16 转换

```
截取高16位: 0x40600000 >> 16 = 0x4060

GFloat16: 0 10000000 1100000
        = [S=0][E=128][M=0.75截断]

还原: 1.75 × 2^1 = 3.5 ✓ (本例无精度损失)
```

### GFloat8 转换

```
截取高8位: 0x40600000 >> 24 = 0x40

GFloat8: 0 1000 000
       = [S=0][E=8][M=0]

还原时指数偏移不同，精度损失较大
```

## 预设格式

| 格式名 | 位宽 | 符号 | 指数 | 尾数 | 精度特点 |
|--------|------|------|------|------|----------|
| `gfloat16` | 16 | 1 | 8 | 7 | 接近 bfloat16，精度较高 |
| `gfloat8` | 8 | 1 | 4 | 3 | 精度很低，仅保留数量级 |

## 与标准格式对比

| 格式 | 位宽 | 指数 | 尾数 | 动态范围 |
|------|------|------|------|----------|
| FP32 | 32 | 8 | 23 | ±3.4×10³⁸ |
| FP16 | 16 | 5 | 10 | ±6.5×10⁴ |
| BF16 | 16 | 8 | 7 | ±3.4×10³⁸ |
| **GFloat16** | 16 | 8 | 7 | ±3.4×10³⁸ |
| **GFloat8** | 8 | 4 | 3 | ±2⁸ |

GFloat16 与 BFloat16 格式相同，保持 FP32 的动态范围。

## 使用方法

### 基本使用

```python
from aidevtools.formats.quantize import quantize
import numpy as np

data = np.array([3.5, 1.2, -2.8, 0.5], dtype=np.float32)

# GFloat16 量化
result16, meta = quantize(data, "gfloat16")
print(result16.dtype)  # uint16

# GFloat8 量化
result8, meta = quantize(data, "gfloat8")
print(result8.dtype)  # uint8
```

### 直接调用 Golden API

```python
from aidevtools.formats.custom.gfloat.golden import to_gfloat16, to_gfloat8

data = np.array([3.5, 1.2, -2.8], dtype=np.float32)

# 量化
gf16, meta = to_gfloat16(data)
gf8, meta = to_gfloat8(data)

print(f"原始: {data}")
print(f"GFloat16: {gf16}")
print(f"GFloat8: {gf8}")
```

## C++ Golden 实现

GFloat 提供 C++ Golden 实现：

```python
from aidevtools.formats.custom.gfloat import register_gfloat_golden

# 注册 C++ golden 实现
register_gfloat_golden()

# 使用 C++ 版本
result, meta = quantize(data, "gfloat16_golden")
```

### 检查 C++ 是否可用

```python
from aidevtools.formats.custom.gfloat import is_cpp_available

if is_cpp_available():
    print("C++ GFloat Golden 可用")
else:
    print("需要编译: cd src/aidevtools/formats/custom/gfloat/cpp && bash build.sh")
```

### C++ 提供的函数

```python
from aidevtools.formats.custom.gfloat.wrapper import (
    fp32_to_gfloat16,
    gfloat16_to_fp32,
    fp32_to_gfloat8,
    gfloat8_to_fp32,
)

# 量化
gf16 = fp32_to_gfloat16(data)

# 反量化
restored = gfloat16_to_fp32(gf16)
```

## 精度特性

### GFloat16

- 与 BFloat16 相同的动态范围
- 尾数精度约 1/128 ≈ 0.8%
- 适合大多数深度学习场景

### GFloat8

- 动态范围极小 (约 ±256)
- 尾数精度约 1/8 ≈ 12.5%
- 仅保留数值的大致数量级
- 适合对精度要求极低的场景

## 实现原理

GFloat 的实现非常简单，就是位截断：

```python
# GFloat16: 截取 FP32 高 16 位
gf16 = (fp32.view(np.uint32) >> 16).astype(np.uint16)

# GFloat8: 截取 FP32 高 8 位
gf8 = (fp32.view(np.uint32) >> 24).astype(np.uint8)

# 反量化: 补零还原
fp32 = (gf16.astype(np.uint32) << 16).view(np.float32)
```

## 与 BFP 对比

| 特性 | GFloat | BFP |
|------|--------|-----|
| 存储方式 | 每元素独立 | 块共享指数 |
| 实现复杂度 | 简单 (位截断) | 中等 (需计算共享指数) |
| 压缩率 | 固定 (16/8 bits) | 可变 (取决于 block_size) |
| 精度特点 | 均匀 | 块内大值精度高 |
| 适用场景 | 通用 | 数据分布集中时更优 |
