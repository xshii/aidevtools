"""Transpose 算子测试"""
import pytest
import numpy as np


class TestTransposePythonGolden:
    """Python Golden 测试"""

    def test_transpose_2d(self):
        """测试 2D 转置"""
        from aidevtools.ops.nn import transpose

        x = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
        y = transpose(x)

        expected = x.T
        np.testing.assert_array_almost_equal(y, expected)

    def test_transpose_4d_default(self):
        """测试 4D 默认转置（交换最后两维）"""
        from aidevtools.ops.nn import transpose

        np.random.seed(42)
        x = np.random.randn(2, 4, 8, 32).astype(np.float32)
        y = transpose(x)

        expected = np.swapaxes(x, -2, -1)
        np.testing.assert_array_almost_equal(y, expected)

    def test_transpose_4d_axes(self):
        """测试 4D 指定轴转置"""
        from aidevtools.ops.nn import transpose

        np.random.seed(42)
        x = np.random.randn(2, 4, 8, 32).astype(np.float32)
        y = transpose(x, axes=(0, 1, 3, 2))

        expected = np.transpose(x, (0, 1, 3, 2))
        np.testing.assert_array_almost_equal(y, expected)

    def test_transpose_reference(self):
        """测试 reference 实现"""
        from aidevtools.ops.nn import Transpose

        np.random.seed(42)
        x = np.random.randn(2, 4, 8, 32).astype(np.float32)

        op = Transpose()
        y = op.reference(x)

        expected = np.swapaxes(x, -2, -1)
        np.testing.assert_array_almost_equal(y, expected)


class TestTransposeCppGolden:
    """C++ Golden 测试"""

    def test_transpose_gfp16(self):
        """测试 gfp16 格式"""
        from aidevtools.ops.cpu_golden import is_cpu_golden_available, set_cpu_golden_dtype
        from aidevtools.ops.nn import Transpose
        from aidevtools.tools.compare.diff import calc_qsnr

        if not is_cpu_golden_available():
            pytest.skip("CPU golden not available")

        set_cpu_golden_dtype("gfp16")
        np.random.seed(42)
        x = np.random.randn(2, 4, 8, 32).astype(np.float32)

        op = Transpose()
        y = op.cpu_golden(x)

        expected = np.swapaxes(x, -2, -1)
        qsnr = calc_qsnr(expected, y)

        assert y.shape == expected.shape
        assert qsnr > 30, f"QSNR {qsnr} dB < 30 dB threshold"

    def test_transpose_gfp8(self):
        """测试 gfp8 格式 (低精度，仅验证形状正确)"""
        from aidevtools.ops.cpu_golden import is_cpu_golden_available, set_cpu_golden_dtype
        from aidevtools.ops.nn import Transpose

        if not is_cpu_golden_available():
            pytest.skip("CPU golden not available")

        set_cpu_golden_dtype("gfp8")
        np.random.seed(42)
        x = np.random.randn(2, 4, 8, 32).astype(np.float32)

        op = Transpose()
        y = op.cpu_golden(x)

        expected = np.swapaxes(x, -2, -1)

        # gfp8 精度很低 (只取 fp32 高 8 位)，仅验证形状
        assert y.shape == expected.shape


class TestTransposeOpsAuto:
    """ops.auto API 测试"""

    def test_transpose_bfp8(self):
        """测试 bfp8 格式"""
        from aidevtools import ops
        from aidevtools.ops.base import get_records

        ops.seed(42)
        ops.clear()

        y = ops.transpose((2, 4, 8, 32), dtype="bfp8")

        records = get_records()
        assert len(records) == 1
        assert records[0]["op"] == "transpose"
        assert y.shape == (2, 4, 32, 8)

    def test_transpose_bfp16(self):
        """测试 bfp16 格式"""
        from aidevtools import ops
        from aidevtools.ops.base import get_records

        ops.seed(42)
        ops.clear()

        y = ops.transpose((2, 4, 8, 32), dtype="bfp16")

        records = get_records()
        assert len(records) == 1
        assert y.shape == (2, 4, 32, 8)

    def test_linear_then_transpose(self):
        """测试 Linear + Transpose 流水线"""
        from aidevtools import ops
        from aidevtools.ops.base import get_records

        ops.seed(42)
        ops.clear()

        y = ops.linear((2, 4, 8, 64), 32, dtype="bfp8")
        y = ops.transpose(y, axes=(0, 1, 3, 2), dtype="bfp8")

        records = get_records()
        assert len(records) == 2
        assert records[0]["op"] == "linear"
        assert records[1]["op"] == "transpose"
        assert y.shape == (2, 4, 32, 8)
