# BFP 实现 TODO

## 概述

BFP (Block Floating Point) 的框架坑位已全部预埋，只需实现量化/反量化算法即可自动接入全链路。

当前状态：
- **BFPP** (Python golden) — 已实现，可作为参考模版
- **BFP** — 坑位已预留，quantize/dequantize 抛 `NotImplementedError`

---

## 唯一必须实现的文件

### `formats/custom/bfp/golden.py`

这是**唯一需要修改的文件**。替换 `_BFP_STUB_FORMATS` 区域的 stub 为真实实现。

参考现有 BFPP 模版 (同文件上方)，需要实现：

```python
# TODO: 实现 BFP 量化函数
def bfp_quantize(data: np.ndarray, block_size, mantissa_bits, **kwargs):
    """
    fp32 → BFP packed int8

    Returns: (packed_data, meta_dict)
    - packed_data: np.ndarray(int8), 格式: [shared_exps | mantissas]
    - meta_dict: 必须包含 block_size, mantissa_bits, num_blocks, original_shape
    """
    ...

# TODO: 实现 BFP 反量化函数
def bfp_dequantize(data: np.ndarray, meta: dict):
    """
    BFP packed int8 → fp32

    Args:
        data: packed_data (与 quantize 返回的格式一致)
        meta: quantize 返回的 meta_dict
    Returns: np.ndarray(float32)
    """
    ...
```

然后用 `register_block_format` 替换 stub 注册：

```python
# 替换 _BFP_STUB_FORMATS 循环为：
for _name, _block_size, _mantissa_bits, _desc in _BFP_STUB_FORMATS:
    register_block_format(BlockFormatSpec(
        name=_name,
        block_size=_block_size,
        mantissa_bits=_mantissa_bits,
        quantize_fn=_make_bfp_quantizer(_block_size, _mantissa_bits),  # 你的实现
        dequantize_fn=_make_bfp_dequantizer(_block_size, _mantissa_bits),  # 你的实现
        description=_desc,
    ))
```

注册后自动接入：
- `quantize(data, "bfp8")`
- `dequantize(packed, "bfp8", meta)`
- `simulate_quantize(data, "bfp8")`
- `load("x.bfp8.bin", shape=...)`
- `compare diff --qtype=bfp8` (bit analysis + blocked)
- `generate_fake_dut(data, "bfp8")`

---

## 自动生效的坑位（无需修改）

以下坑位在 `register_block_format` 调用后自动生效：

| 文件 | 坑位 | 状态 |
|------|------|------|
| `formats/block_format.py` | registry 查询 API | 已就绪 |
| `formats/quantize.py` | `_quantize_registry` / `_dequantize_registry` | 自动注册 |
| `formats/base.py` | `load()` block format 分支 | 已就绪 |
| `commands/compare.py` | `get_bit_layout()` / `get_block_format()` | 已就绪 |
| `compare/strategy/bit_analysis.py` | `BFP16` / `BFP8` / `BFP4` BitLayout 常量 | 已定义 |
| `formats/filename_parser.py` | `_KNOWN_QTYPES` 包含 bfp16/bfp8/bfp4 | 已预留 |
| `frontend/types.py` | `DType.BFP16` / `DType.BFP8` 枚举 | 已定义 |
| `core/constants.py` | `BFP_TYPES` / `BFP*_BLOCK_SIZE` 常量 | 已定义 |
| `core/memory_types.py` | `dtype_sizes` 包含 bfp16/bfp8 | 已定义 |
| `datagen.py` | `hw_qtypes` 包含 bfp16/bfp8/bfp4 | 已就绪 |

---

## 独立的 C++ Golden（可选）

C++ Golden 是独立的硬件仿真实现，与 Python `register_block_format` 无关。

| 文件 | 说明 | 状态 |
|------|------|------|
| `golden/cpp/bfp/io.h` | BFPType 枚举 + 读写接口 | 已实现 |
| `golden/cpp/bfp/io.cpp` | BFP 文件 I/O | 已实现 |
| `golden/cpp/bfp/impl.cpp` | matmul/softmax/layernorm 等 12 个 golden | 已实现 |
| `golden/cpp/bfp_main.cpp` | CLI 入口 | 已实现 |
| `formats/custom/bfp/wrapper.py` | Python 调用 C++ 的包装器 | 已实现 |
| `ops/cpu_golden.py` | `_is_bfp_type()` / `_get_bfp_params()` | 已实现 |
| `ops/traced_tensor.py` | `BFPType` / `_is_bfp_type()` | 已实现 |

> C++ Golden 通过 `qtype="bfp16_golden"` 调用，与 `register_block_format("bfp16")` 是两套独立路径。

---

## 验证

实现后运行：

```bash
# 全量测试
python -m pytest tests/ut/ -v

# BFP 专项
python -m pytest tests/ut/test_bfp.py tests/ut/formats/test_block_format.py -v

# 快速验证
python -c "
from aidevtools.formats.quantize import simulate_quantize
import numpy as np
data = np.random.randn(64).astype(np.float32)
result = simulate_quantize(data, 'bfp8')
print('bfp8 roundtrip OK, max_diff:', np.max(np.abs(data - result)))
"
```
