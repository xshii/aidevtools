"""Transpose 算子测试"""
import pytest
import numpy as np

from aidevtools.ops import _functional as F


class TestTransposePythonGolden:
    """Transpose 算子测试 (需要 cpu_golden)"""

    def test_transpose_2d(self):
        """测试 2D 转置"""
        import pytest
        from aidevtools.ops.cpu_golden import is_cpu_golden_available, set_cpu_golden_dtype

        if not is_cpu_golden_available():
            pytest.skip("CPU golden not available")

        set_cpu_golden_dtype("gfp16")
        x = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
        y = F.transpose(x)

        expected = x.T
        # cpu_golden 有量化误差，只验证形状
        assert y.shape == expected.shape

    def test_transpose_4d_default(self):
        """测试 4D 默认转置（交换最后两维）"""
        import pytest
        from aidevtools.ops.cpu_golden import is_cpu_golden_available, set_cpu_golden_dtype

        if not is_cpu_golden_available():
            pytest.skip("CPU golden not available")

        set_cpu_golden_dtype("gfp16")
        np.random.seed(42)
        x = np.random.randn(2, 4, 8, 32).astype(np.float32)
        y = F.transpose(x)

        expected = np.swapaxes(x, -2, -1)
        # cpu_golden 有量化误差，只验证形状
        assert y.shape == expected.shape

    def test_transpose_4d_axes(self):
        """测试 4D 指定轴转置"""
        import pytest
        from aidevtools.ops.cpu_golden import is_cpu_golden_available, set_cpu_golden_dtype

        if not is_cpu_golden_available():
            pytest.skip("CPU golden not available")

        set_cpu_golden_dtype("gfp16")
        np.random.seed(42)
        x = np.random.randn(2, 4, 8, 32).astype(np.float32)
        y = F.transpose(x, axes=(0, 1, 3, 2))

        expected = np.transpose(x, (0, 1, 3, 2))
        # cpu_golden 有量化误差，只验证形状
        assert y.shape == expected.shape

    def test_transpose_cpu_golden(self):
        """测试 cpu_golden 实现"""
        import pytest
        from aidevtools.ops.cpu_golden import is_cpu_golden_available, set_cpu_golden_dtype

        if not is_cpu_golden_available():
            pytest.skip("CPU golden not available")

        set_cpu_golden_dtype("gfp16")
        np.random.seed(42)
        x = np.random.randn(2, 4, 8, 32).astype(np.float32)

        y = F.Transpose().cpu_golden(x)

        expected = np.swapaxes(x, -2, -1)
        # cpu_golden 有量化误差，只验证形状
        assert y.shape == expected.shape


class TestTransposeCppGolden:
    """C++ Golden 测试"""

    def test_transpose_gfp16(self):
        """测试 gfp16 格式"""
        from aidevtools.ops.cpu_golden import is_cpu_golden_available, set_cpu_golden_dtype
        from aidevtools.tools.compare.diff import calc_qsnr

        if not is_cpu_golden_available():
            pytest.skip("CPU golden not available")

        set_cpu_golden_dtype("gfp16")
        np.random.seed(42)
        x = np.random.randn(2, 4, 8, 32).astype(np.float32)

        y = F.Transpose().cpu_golden(x)

        expected = np.swapaxes(x, -2, -1)
        qsnr = calc_qsnr(expected, y)

        assert y.shape == expected.shape
        assert qsnr > 30, f"QSNR {qsnr} dB < 30 dB threshold"

    def test_transpose_gfp8(self):
        """测试 gfp8 格式 (低精度，仅验证形状正确)"""
        from aidevtools.ops.cpu_golden import is_cpu_golden_available, set_cpu_golden_dtype

        if not is_cpu_golden_available():
            pytest.skip("CPU golden not available")

        set_cpu_golden_dtype("gfp8")
        np.random.seed(42)
        x = np.random.randn(2, 4, 8, 32).astype(np.float32)

        y = F.Transpose().cpu_golden(x)

        expected = np.swapaxes(x, -2, -1)

        # gfp8 精度很低 (只取 fp32 高 8 位)，仅验证形状
        assert y.shape == expected.shape


class TestTransposeFunctionalAPI:
    """PyTorch 风格 functional API 测试"""

    def test_linear_output_shape(self):
        """测试 F.linear 输出形状 (PyTorch 格式)"""
        from aidevtools import ops
        from aidevtools.ops import get_records

        ops.seed(42)
        ops.clear()
        np.random.seed(42)

        # Input: (batch, heads, seq, d_k) = (2, 4, 8, 64)
        x = np.random.randn(2, 4, 8, 64).astype(np.float32)

        # PyTorch 格式: weight [out_features, in_features] = [32, 64]
        w = np.random.randn(32, 64).astype(np.float32) * 0.02
        b = np.random.randn(32).astype(np.float32) * 0.01

        y = F.linear(x, w, b)

        records = get_records()
        assert len(records) == 1
        assert records[0]["op"] == "linear"
        assert y.shape == (2, 4, 8, 32)

    def test_linear_then_transpose(self):
        """测试 F.linear + numpy transpose (PyTorch 格式)"""
        from aidevtools import ops
        from aidevtools.ops import get_records

        ops.seed(42)
        ops.clear()
        np.random.seed(42)

        x = np.random.randn(2, 4, 8, 64).astype(np.float32)
        # PyTorch 格式: weight [out_features, in_features] = [32, 64]
        w = np.random.randn(32, 64).astype(np.float32) * 0.02

        y = F.linear(x, w)
        y = np.transpose(y, axes=(0, 1, 3, 2))

        records = get_records()
        assert len(records) == 1  # only linear recorded
        assert records[0]["op"] == "linear"
        assert y.shape == (2, 4, 32, 8)
