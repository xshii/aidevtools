# Block Format 注册框架 — 新增量化格式适配指南

## 概述

通过 `register_block_format()` 一次注册，自动接入以下全链路：
- **quantize()** — 量化
- **dequantize()** — 反量化
- **simulate_quantize()** — 模拟量化精度损失
- **load()** — 从文件加载并自动反量化
- **compare diff** — bit analysis + blocked 分析
- **generate_fake_dut()** — 生成模拟 DUT 数据

## 适配步骤

### 1. 实现量化/反量化函数

```python
import numpy as np

def my_quantize(data: np.ndarray, **kwargs) -> tuple:
    """
    量化: fp32 → packed int8

    Args:
        data: 输入数据 (fp32)
        **kwargs: 可选参数 (如 block_size 覆盖)

    Returns:
        (packed_data, meta_dict)
        - packed_data: np.ndarray, dtype 应与 storage_dtype 一致
        - meta_dict: 必须包含以下 key:
            - "block_size": int
            - "mantissa_bits": int
            - "num_blocks": int
            - "original_shape": tuple
          可选 key:
            - "format": str  (格式名)
            - 任意自定义 key
    """
    block_size = kwargs.get("block_size", 32)
    # ... 你的量化算法 ...
    packed = np.concatenate([shared_exps, mantissas])  # 示例打包格式
    meta = {
        "format": "bfp",
        "block_size": block_size,
        "mantissa_bits": 4,
        "num_blocks": num_blocks,
        "original_shape": data.shape,
    }
    return packed, meta


def my_dequantize(data: np.ndarray, meta: dict) -> np.ndarray:
    """
    反量化: packed int8 → fp32

    Args:
        data: 打包后的量化数据
        meta: 量化时返回的 meta_dict

    Returns:
        fp32 还原数据, shape 应恢复为 meta["original_shape"]
    """
    num_blocks = meta["num_blocks"]
    block_size = meta["block_size"]
    # ... 你的反量化算法 ...
    return result.reshape(meta["original_shape"]).astype(np.float32)
```

### 2. 注册格式

```python
from aidevtools.formats.block_format import BlockFormatSpec, register_block_format

register_block_format(BlockFormatSpec(
    name="bfp8",                    # 格式名（用于 qtype 参数）
    block_size=32,                  # 每块元素数
    mantissa_bits=4,                # 有效位数（含符号位）
    quantize_fn=my_quantize,        # 量化函数
    dequantize_fn=my_dequantize,    # 反量化函数
    storage_dtype=np.int8,          # 存储类型 (默认 np.int8)
    description="BFP8 custom impl", # 可选描述
))
```

### 3. 注册时机

在 `aidevtools/formats/custom/bfp/golden.py` 底部（或新建模块），确保在
`aidevtools/formats/__init__.py` 的 **第 4 步**（加载自定义格式）中被导入。

导入顺序：
```
1. base        → FormatBase
2. numpy/raw   → 内置格式
3. quantize    → 量化模块
3.5 block_format → 注册框架
4. custom/...  → ← 你的注册代码在这里
```

### 4. 文件名解析支持

如果需要从文件名自动推断 qtype（如 `softmax_bfp8_2x16x64.txt`），
需要在 `aidevtools/formats/filename_parser.py` 的 `_KNOWN_QTYPES` 列表中
添加你的格式名：

```python
_KNOWN_QTYPES = [
    ...
    "bfp16", "bfp8", "bfp4",       # ← 已预留
    ...
]
```

> 当前 bfp16/bfp8/bfp4 已预留在列表中，注册后即可使用。

## 自动接入链路说明

| 链路 | 触发方式 | 说明 |
|------|----------|------|
| `quantize(data, "bfp8")` | 自动 | register_block_format 注册到 _quantize_registry |
| `dequantize(packed, "bfp8", meta)` | 自动 | register_block_format 注册到 _dequantize_registry |
| `simulate_quantize(data, "bfp8")` | 自动 | 内部调用 quantize + dequantize |
| `load("x.bfp8.bin", shape=...)` | 自动 | base.py 检查 is_block_format() → 用 spec 加载 |
| `compare diff --qtype=bfp8` | 自动 | compare.py 检查 is_block_format() → 用 registry 获取 bit_layout 和 block_size |
| `_infer_qtype("x.bfp8.bin")` | 需配置 | 需在 filename_parser._KNOWN_QTYPES 中注册 |

## 查询 API

```python
from aidevtools.formats.block_format import (
    is_block_format,     # 判断是否已注册
    get_block_format,    # 获取 BlockFormatSpec
    list_block_formats,  # 列出所有已注册格式
    get_bit_layout,      # 自动生成 BitLayout（用于 bit analysis）
)

is_block_format("bfp8")       # True (注册后)
get_block_format("bfp8")      # BlockFormatSpec(...)
list_block_formats()           # ["bfpp16", "bfpp8", "bfpp4", "bfp8", ...]
get_bit_layout("bfp8")        # BitLayout(1, 0, 7, "bfp8", precision_bits=4)
```

## 现有格式参考

现有 BFPP（纯 Python 实现）的注册代码在 `formats/custom/bfp/golden.py` 底部：

```python
for _name, _block_size, _mantissa_bits, _desc in _BFP_FORMATS:
    register_block_format(BlockFormatSpec(
        name=_name,
        block_size=_block_size,
        mantissa_bits=_mantissa_bits,
        quantize_fn=_make_bfp_quantizer(_block_size, _mantissa_bits),
        dequantize_fn=_make_bfp_dequantizer(_block_size, _mantissa_bits),
        description=_desc,
    ))
```
