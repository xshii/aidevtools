# 添加新算子指南

以 RMSNorm 为例，说明添加一个新算子需要修改哪些文件。

## 文件清单

| 文件 | 是否必须 | 说明 |
|------|---------|------|
| `src/aidevtools/ops/_functional.py` | ✅ 必须 | 添加算子类 |
| `src/aidevtools/ops/auto.py` | 🔄 自动 | 基于 `auto_gen` 元数据自动生成，无需修改 |
| `src/aidevtools/golden/cpp/` | 可选 | 添加 C++ Golden |
| `tests/ut/test_*.py` | ✅ 必须 | 添加单元测试 |
| `src/aidevtools/xlsx/op_registry.py` | 可选 | xlsx 额外算子 |

---

## Step 1: 添加算子类 (`ops/_functional.py`)

```python
# src/aidevtools/ops/_functional.py

@register_op(
    inputs=["x", "gamma"],           # 必需输入参数
    optional=["eps"],                # 可选参数
    description="RMS Normalization",
    has_cpp_golden=False,            # 是否有 C++ Golden (Step 3)
    auto_gen={                       # 简化 API 参数生成策略
        "x": "input",                # 主输入 (shape 或 array)
        "gamma": "ones:-1",          # 全1数组，shape 取输入最后一维
    },
)
class RMSNorm(Op):
    """RMS Normalization: y = x / rms(x) * gamma"""
    name = "rmsnorm"

    def golden_python(self, x: np.ndarray, gamma: np.ndarray, eps: float = 1e-5) -> np.ndarray:
        """Python Golden 实现 (fp32)"""
        rms = np.sqrt(np.mean(x ** 2, axis=-1, keepdims=True) + eps)
        return (x / rms * gamma).astype(np.float32)

    def reference(self, x: np.ndarray, gamma: np.ndarray, eps: float = 1e-5) -> np.ndarray:
        """高精度参考实现 (fp64)"""
        x64 = x.astype(np.float64)
        gamma64 = gamma.astype(np.float64)
        rms = np.sqrt(np.mean(x64 ** 2, axis=-1, keepdims=True) + eps)
        return (x64 / rms * gamma64).astype(np.float32)

    # 如果 has_cpp_golden=True，还需添加:
    # def cpu_golden(self, x: np.ndarray, gamma: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    #     """C++ Golden 实现"""
    #     ...


# 文件末尾添加实例
rmsnorm = RMSNorm()
```

### `@register_op` 参数说明

| 参数 | 类型 | 说明 |
|------|------|------|
| `inputs` | `List[str]` | 必需输入参数名列表 |
| `optional` | `List[str]` | 可选参数名列表 |
| `description` | `str` | 算子描述 |
| `has_cpp_golden` | `bool` | 是否有 C++ Golden 实现 |
| `auto_gen` | `Dict[str, str]` | 简化 API 参数生成策略 |

### `auto_gen` 策略说明

| 策略 | 说明 | 示例 |
|------|------|------|
| `"input"` | 主输入，可以是 shape 或 array | 第一个参数 |
| `"random"` | 随机初始化，shape 与输入相同 | 默认策略 |
| `"ones:-1"` | 全1数组，-1 表示取输入最后一维 | gamma |
| `"zeros:-1"` | 全0数组 | beta, bias |
| `"xavier"` | Xavier 初始化 (用于 weight) | linear weight |

---

## Step 2: 简化 API (自动生成，无需修改)

配置了 `auto_gen` 后，`ops.rmsnorm(shape, ...)` 会**自动可用**，无需修改 `auto.py`。

```python
# 使用示例 - 无需任何额外代码
from aidevtools import ops

ops.seed(42)
y = ops.rmsnorm((2, 8, 64))  # 自动生成 gamma=1
```

**工作原理：**
- `auto.py` 通过 `__getattr__` 动态获取任何已注册的算子
- 根据 `auto_gen` 配置自动生成参数 (gamma=ones)
- 如果没有配置 `auto_gen`，默认策略：第一个输入为 `"input"`，其他为 `"random"`

**只有复杂算子需要手动添加**（如 `linear`, `attention`），因为它们需要额外参数（`out_features`, `mask` 等）。

---

## Step 3: 添加 C++ Golden [可选]

如果需要 C++ Golden 实现 (用于 gfloat 格式):

### 3.1 修改 C++ 源码

```cpp
// src/aidevtools/golden/cpp/cpu_golden.cpp

// 添加 rmsnorm 实现
void rmsnorm(const std::string& dtype, ...) {
    // 实现 RMS Normalization
}

// 在 main() 中添加分支
if (op == "rmsnorm") {
    rmsnorm(dtype, ...);
}
```

### 3.2 重新编译

```bash
cd src/aidevtools/golden/cpp
./build.sh
```

### 3.3 添加 `cpu_golden` 方法

```python
# src/aidevtools/ops/_functional.py - RMSNorm 类中添加

def cpu_golden(self, x: np.ndarray, gamma: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """C++ Golden 实现"""
    from aidevtools.ops.cpu_golden import run_cpu_golden, get_cpu_golden_dtype

    dtype = get_cpu_golden_dtype()
    x = np.asarray(x, dtype=np.float32)
    gamma = np.asarray(gamma, dtype=np.float32)

    original_shape = x.shape
    hidden = x.shape[-1]

    # flatten 到 2D
    if x.ndim == 1:
        x = x.reshape(1, -1)
    elif x.ndim > 2:
        x = x.reshape(-1, hidden)

    batch = x.shape[0]

    y = run_cpu_golden(
        op_name="rmsnorm",
        cmd_args=["rmsnorm", dtype, "@x.bin", "@gamma.bin", "@output", str(batch), str(hidden)],
        inputs={
            "x.bin": (x, dtype),
            "gamma.bin": (gamma, dtype),
        },
        output_name="y.bin",
        output_dtype=dtype,
        output_size=batch * hidden,
        output_shape=(batch, hidden),
    )

    return y.reshape(original_shape)
```

### 3.4 更新 `@register_op`

```python
@register_op(
    inputs=["x", "gamma"],
    optional=["eps"],
    description="RMS Normalization",
    has_cpp_golden=True,  # 改为 True
)
class RMSNorm(Op):
    ...
```

---

## Step 4: 添加单元测试

```python
# tests/ut/test_rmsnorm.py

import pytest
import numpy as np
from aidevtools.ops import _functional as F


class TestRMSNormPythonGolden:
    """Python Golden 测试"""

    def test_rmsnorm_basic(self):
        """基本功能测试"""
        x = np.random.randn(2, 8, 64).astype(np.float32)
        gamma = np.ones(64, dtype=np.float32)

        y = F.rmsnorm(x, gamma)

        assert y.shape == x.shape
        assert y.dtype == np.float32

    def test_rmsnorm_reference(self):
        """reference 实现测试"""
        x = np.random.randn(2, 64).astype(np.float32)
        gamma = np.ones(64, dtype=np.float32)

        y = F.RMSNorm().reference(x, gamma)

        # 验证 RMS 归一化后的值
        assert y.shape == x.shape


class TestRMSNormCppGolden:
    """C++ Golden 测试 (如果有)"""

    def test_rmsnorm_gfp16(self):
        """gfp16 格式测试"""
        from aidevtools.ops.cpu_golden import is_cpu_golden_available, set_cpu_golden_dtype

        if not is_cpu_golden_available():
            pytest.skip("CPU golden not available")

        set_cpu_golden_dtype("gfp16")

        x = np.random.randn(2, 64).astype(np.float32)
        gamma = np.ones(64, dtype=np.float32)

        y = F.RMSNorm().cpu_golden(x, gamma)

        assert y.shape == x.shape
```

---

## Step 5: xlsx 支持 [可选]

如果需要在 xlsx 中支持该算子:

```python
# src/aidevtools/xlsx/op_registry.py

# 在 EXTRA_OPS 中添加
EXTRA_OPS = [
    "conv2d",
    "pooling",
    "rmsnorm",  # 新增
]
```

---

## 完整检查清单

添加新算子时，检查以下项目:

- [ ] `ops/_functional.py` - 添加算子类，包含 `golden_python` 和 `reference` 方法
- [ ] `ops/_functional.py` - 配置 `@register_op` 的 `auto_gen` 参数
- [ ] `ops/_functional.py` - 文件末尾添加实例 (如 `rmsnorm = RMSNorm()`)
- [ ] `ops/auto.py` - 🔄 **自动生成**，普通算子无需修改
- [ ] `golden/cpp/` - 添加 C++ 实现并重新编译 (可选)
- [ ] `ops/_functional.py` - 添加 `cpu_golden` 方法 (如果有 C++ Golden)
- [ ] `tests/ut/` - 添加单元测试
- [ ] `xlsx/op_registry.py` - 添加到 EXTRA_OPS (可选)

---

## 运行测试

```bash
# 运行所有测试
pytest tests/ut/ -v

# 只运行新算子测试
pytest tests/ut/test_rmsnorm.py -v
```
