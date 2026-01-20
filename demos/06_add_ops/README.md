# 添加新算子指南

以 RMSNorm 为例，说明添加一个新算子需要修改哪些文件。

## 文件清单

| 文件 | 是否必须 | 说明 |
|------|---------|------|
| `src/aidevtools/ops/nn.py` | ✅ 必须 | 添加算子类 |
| `src/aidevtools/ops/auto.py` | 可选 | 添加简化 API |
| `src/aidevtools/golden/cpp/` | 可选 | 添加 C++ Golden |
| `tests/ut/test_*.py` | ✅ 必须 | 添加单元测试 |
| `src/aidevtools/xlsx/op_registry.py` | 可选 | xlsx 额外算子 |

---

## Step 1: 添加算子类 (`ops/nn.py`)

```python
# src/aidevtools/ops/nn.py

@register_op(
    inputs=["x", "gamma"],           # 必需输入参数
    optional=["eps"],                # 可选参数
    description="RMS Normalization",
    has_cpp_golden=False,            # 是否有 C++ Golden (Step 3)
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

---

## Step 2: 添加简化 API (`ops/auto.py`) [可选]

如果需要支持 `ops.rmsnorm(shape, ...)` 自动生成数据的用法:

```python
# src/aidevtools/ops/auto.py

def rmsnorm(
    x: np.ndarray,
    normalized_shape: int,
    eps: float = 1e-5,
    dtype: str = DEFAULT_DTYPE,
) -> np.ndarray:
    """
    RMSNorm 层

    Args:
        x: 输入数据
        normalized_shape: 归一化维度大小
        eps: epsilon
        dtype: 数据类型

    Returns:
        输出数据
    """
    # gamma 初始化为 1
    gamma = np.ones(normalized_shape, dtype=np.float32)

    # 量化
    x = _quantize_input(x, dtype)
    gamma = _quantize_input(gamma, dtype)

    return _nn.rmsnorm(x, gamma, eps)
```

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
# src/aidevtools/ops/nn.py - RMSNorm 类中添加

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


class TestRMSNormPythonGolden:
    """Python Golden 测试"""

    def test_rmsnorm_basic(self):
        """基本功能测试"""
        from aidevtools.ops.nn import rmsnorm

        x = np.random.randn(2, 8, 64).astype(np.float32)
        gamma = np.ones(64, dtype=np.float32)

        y = rmsnorm(x, gamma)

        assert y.shape == x.shape
        assert y.dtype == np.float32

    def test_rmsnorm_reference(self):
        """reference 实现测试"""
        from aidevtools.ops.nn import RMSNorm

        x = np.random.randn(2, 64).astype(np.float32)
        gamma = np.ones(64, dtype=np.float32)

        op = RMSNorm()
        y = op.reference(x, gamma)

        # 验证 RMS 归一化后的值
        assert y.shape == x.shape


class TestRMSNormCppGolden:
    """C++ Golden 测试 (如果有)"""

    def test_rmsnorm_gfp16(self):
        """gfp16 格式测试"""
        from aidevtools.ops.cpu_golden import is_cpu_golden_available, set_cpu_golden_dtype
        from aidevtools.ops.nn import RMSNorm

        if not is_cpu_golden_available():
            pytest.skip("CPU golden not available")

        set_cpu_golden_dtype("gfp16")

        x = np.random.randn(2, 64).astype(np.float32)
        gamma = np.ones(64, dtype=np.float32)

        op = RMSNorm()
        y = op.cpu_golden(x, gamma)

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

- [ ] `ops/nn.py` - 添加算子类，包含 `golden_python` 和 `reference` 方法
- [ ] `ops/nn.py` - 文件末尾添加实例 (如 `rmsnorm = RMSNorm()`)
- [ ] `ops/auto.py` - 添加简化 API (可选)
- [ ] `golden/cpp/` - 添加 C++ 实现并重新编译 (可选)
- [ ] `ops/nn.py` - 添加 `cpu_golden` 方法 (如果有 C++ Golden)
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
