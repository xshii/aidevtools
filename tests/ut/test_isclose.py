"""IsClose 比对测试"""
import pytest
import numpy as np
from aidevtools.tools.compare.diff import compare_isclose, IsCloseResult


class TestIsClose:
    """IsClose 比对测试"""

    def test_perfect_match(self):
        """完美匹配应该通过"""
        golden = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        result = golden.copy()

        r = compare_isclose(golden, result, atol=1e-5, rtol=1e-3, max_exceed_ratio=0.0)

        assert r.passed is True
        assert r.exceed_count == 0
        assert r.exceed_ratio == 0.0
        assert r.max_abs_error == 0.0

    def test_small_noise_pass(self):
        """小噪声应该通过"""
        np.random.seed(42)
        golden = np.random.randn(100).astype(np.float32)
        result = golden + np.random.randn(100).astype(np.float32) * 1e-5

        r = compare_isclose(golden, result, atol=1e-4, rtol=1e-2, max_exceed_ratio=0.01)

        assert r.passed is True

    def test_large_noise_fail(self):
        """大噪声应该失败"""
        np.random.seed(42)
        golden = np.random.randn(100).astype(np.float32)
        result = golden + np.random.randn(100).astype(np.float32) * 0.5

        r = compare_isclose(golden, result, atol=1e-4, rtol=1e-2, max_exceed_ratio=0.01)

        assert r.passed is False
        assert r.exceed_ratio > 0.5  # 大部分元素应该超限

    def test_exceed_ratio_threshold(self):
        """测试超限比例门限"""
        np.random.seed(42)
        golden = np.ones(100, dtype=np.float32)
        # 让 10% 的元素有大误差
        result = golden.copy()
        result[:10] = golden[:10] + 0.5

        # 允许 5% 超限 -> 失败
        r1 = compare_isclose(golden, result, atol=1e-4, rtol=1e-2, max_exceed_ratio=0.05)
        assert r1.passed is False

        # 允许 15% 超限 -> 通过
        r2 = compare_isclose(golden, result, atol=1e-4, rtol=1e-2, max_exceed_ratio=0.15)
        assert r2.passed is True

    def test_atol_only(self):
        """测试仅绝对误差门限"""
        golden = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        result = np.array([1e-5, 1e-5, 1e-5], dtype=np.float32)

        # atol=1e-4 -> 通过
        r1 = compare_isclose(golden, result, atol=1e-4, rtol=0.0, max_exceed_ratio=0.0)
        assert r1.passed is True

        # atol=1e-6 -> 失败
        r2 = compare_isclose(golden, result, atol=1e-6, rtol=0.0, max_exceed_ratio=0.0)
        assert r2.passed is False

    def test_rtol_only(self):
        """测试仅相对误差门限"""
        golden = np.array([100.0, 100.0, 100.0], dtype=np.float32)
        result = np.array([100.1, 100.1, 100.1], dtype=np.float32)  # 0.1% 相对误差

        # rtol=0.01 (1%) -> 通过
        r1 = compare_isclose(golden, result, atol=0.0, rtol=0.01, max_exceed_ratio=0.0)
        assert r1.passed is True

        # rtol=0.0001 (0.01%) -> 失败
        r2 = compare_isclose(golden, result, atol=0.0, rtol=0.0001, max_exceed_ratio=0.0)
        assert r2.passed is False

    def test_fp32_conversion(self):
        """测试自动转换为 fp32"""
        golden = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        result = np.array([1.0, 2.0, 3.0], dtype=np.float16)

        r = compare_isclose(golden, result, atol=1e-5, rtol=1e-3, max_exceed_ratio=0.0)

        assert r.passed is True
        assert r.total_elements == 3

    def test_shape_mismatch(self):
        """测试 shape 不匹配"""
        golden = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        result = np.array([1.0, 2.0], dtype=np.float32)

        with pytest.raises(ValueError, match="Shape mismatch"):
            compare_isclose(golden, result)

    def test_result_dataclass(self):
        """测试结果数据类的字段"""
        golden = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        result = np.array([1.1, 2.0, 3.0], dtype=np.float32)

        r = compare_isclose(golden, result, atol=0.05, rtol=0.01, max_exceed_ratio=0.5)

        assert isinstance(r, IsCloseResult)
        assert r.total_elements == 3
        assert r.atol == 0.05
        assert r.rtol == 0.01
        assert r.max_exceed_ratio == 0.5
        assert r.max_abs_error > 0
        assert r.mean_abs_error > 0
