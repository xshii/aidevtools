"""TracedTensor 单元测试"""
import numpy as np
import pytest

from aidevtools.ops.traced_tensor import (
    TracedTensor,
    traced,
    ensure_traced,
    wrap_traced_output,
)


class TestTracedTensor:
    """TracedTensor 类测试"""

    def test_create_basic(self):
        """创建基本 TracedTensor"""
        data = np.array([[1, 2], [3, 4]], dtype=np.float32)
        t = TracedTensor(data=data)

        assert t.shape == (2, 2)
        assert t.size == 4
        assert t.ndim == 2
        assert t.dtype is None
        assert t.source_op is None
        assert not t.is_weight
        assert not t.is_quantized

    def test_create_with_dtype(self):
        """创建带 dtype 的 TracedTensor"""
        data = np.array([1, 2, 3], dtype=np.float32)
        t = TracedTensor(data=data, dtype="gfp16")

        assert t.dtype == "gfp16"
        assert t.is_quantized

    def test_create_with_source(self):
        """创建带 source_op 的 TracedTensor"""
        data = np.array([1, 2, 3], dtype=np.float32)
        t = TracedTensor(data=data, source_op="matmul_0")

        assert t.source_op == "matmul_0"

    def test_create_weight(self):
        """创建权重 TracedTensor"""
        data = np.array([1, 2, 3], dtype=np.float32)
        t = TracedTensor(data=data, is_weight=True)

        assert t.is_weight

    def test_numpy(self):
        """获取 numpy 数组"""
        data = np.array([[1, 2], [3, 4]], dtype=np.float32)
        t = TracedTensor(data=data)

        result = t.numpy()
        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, data)

    def test_auto_convert_to_fp32(self):
        """自动转换为 fp32"""
        data = np.array([1, 2, 3], dtype=np.float16)
        t = TracedTensor(data=data)

        assert t.data.dtype == np.float32

    def test_with_source(self):
        """with_source 方法"""
        data = np.array([1, 2, 3], dtype=np.float32)
        t = TracedTensor(data=data, dtype="gfp16")

        t2 = t.with_source("gelu_0")
        assert t2.source_op == "gelu_0"
        assert t2.dtype == "gfp16"  # 保留原有 dtype
        np.testing.assert_array_equal(t2.data, t.data)

    def test_reshape(self):
        """reshape 方法"""
        data = np.array([[1, 2], [3, 4]], dtype=np.float32)
        t = TracedTensor(data=data, dtype="gfp16", source_op="test")

        t2 = t.reshape(4)
        assert t2.shape == (4,)
        assert t2.dtype == "gfp16"
        assert t2.source_op == "test"

    def test_transpose(self):
        """transpose 方法"""
        data = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
        t = TracedTensor(data=data, dtype="gfp16")

        t2 = t.transpose()
        assert t2.shape == (3, 2)
        assert t2.dtype == "gfp16"

    def test_getitem(self):
        """切片操作"""
        data = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float32)
        t = TracedTensor(data=data, dtype="gfp16")

        t2 = t[0]
        assert t2.shape == (2,)
        np.testing.assert_array_equal(t2.data, [1, 2])

        t3 = t[:2]
        assert t3.shape == (2, 2)

    def test_flatten(self):
        """flatten 方法"""
        data = np.array([[1, 2], [3, 4]], dtype=np.float32)
        t = TracedTensor(data=data, dtype="gfp16")

        t2 = t.flatten()
        assert t2.shape == (4,)
        assert t2.dtype == "gfp16"

    def test_repr(self):
        """__repr__ 方法"""
        t = TracedTensor(
            data=np.array([1, 2, 3], dtype=np.float32),
            dtype="gfp16",
            source_op="matmul_0",
            is_weight=True,
        )

        s = repr(t)
        assert "TracedTensor" in s
        assert "gfp16" in s
        assert "matmul_0" in s
        assert "weight=True" in s


class TestTracedFunction:
    """traced 工厂函数测试"""

    def test_traced_from_numpy(self):
        """从 numpy 数组创建"""
        data = np.array([1, 2, 3], dtype=np.float32)
        t = traced(data)

        assert isinstance(t, TracedTensor)
        assert t.dtype is None
        assert not t.is_weight

    def test_traced_with_dtype(self):
        """创建带 dtype 的 TracedTensor"""
        from aidevtools.ops.cpu_golden import is_cpu_golden_available

        if not is_cpu_golden_available():
            pytest.skip("CPU golden not available for quantization")

        data = np.array([1, 2, 3], dtype=np.float32)
        t = traced(data, "gfp16")

        assert t.dtype == "gfp16"

    def test_traced_weight(self):
        """创建权重"""
        data = np.array([1, 2, 3], dtype=np.float32)
        t = traced(data, is_weight=True)

        assert t.is_weight

    def test_traced_from_traced_tensor(self):
        """从 TracedTensor 创建"""
        data = np.array([1, 2, 3], dtype=np.float32)
        t1 = traced(data)
        t2 = traced(t1, is_weight=True)

        assert t2.is_weight


class TestWrapTracedOutput:
    """wrap_traced_output 函数测试"""

    def test_wrap_output(self):
        """包装算子输出"""
        data = np.array([1, 2, 3], dtype=np.float32)
        t = wrap_traced_output(data, "gfp16", "relu_0")

        assert isinstance(t, TracedTensor)
        assert t.dtype == "gfp16"
        assert t.source_op == "relu_0"
        assert not t.is_weight  # 输出不是权重


class TestEnsureTraced:
    """ensure_traced 函数测试"""

    def test_ensure_traced_from_numpy(self):
        """从 numpy 确保 TracedTensor"""
        from aidevtools.ops.cpu_golden import is_cpu_golden_available

        if not is_cpu_golden_available():
            pytest.skip("CPU golden not available for quantization")

        data = np.array([1, 2, 3], dtype=np.float32)

        with pytest.warns(UserWarning, match="raw numpy array"):
            t = ensure_traced(data, "gfp16")

        assert t.dtype == "gfp16"

    def test_ensure_traced_from_traced_tensor(self):
        """从 TracedTensor 确保精度"""
        from aidevtools.ops.cpu_golden import is_cpu_golden_available

        if not is_cpu_golden_available():
            pytest.skip("CPU golden not available for quantization")

        data = np.array([1, 2, 3], dtype=np.float32)
        t1 = traced(data, "gfp16")

        # 已经是目标精度，直接返回
        t2 = ensure_traced(t1, "gfp16")
        assert t2 is t1

    def test_ensure_traced_no_warn(self):
        """禁用警告"""
        from aidevtools.ops.cpu_golden import is_cpu_golden_available

        if not is_cpu_golden_available():
            pytest.skip("CPU golden not available for quantization")

        data = np.array([1, 2, 3], dtype=np.float32)
        t = ensure_traced(data, "gfp16", warn=False)

        assert t.dtype == "gfp16"
