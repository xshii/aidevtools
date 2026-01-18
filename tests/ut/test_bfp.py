"""BFP (Block Floating Point) 单元测试"""
import pytest
import numpy as np

from aidevtools.formats.custom.bfp.golden import fp32_to_bfp, bfp_to_fp32


class TestBfpPython:
    """Python BFP 实现测试"""

    def test_fp32_to_bfp_basic(self):
        """基本量化测试"""
        data = np.array([1.0, 2.0, -3.0, 0.5], dtype=np.float32)
        mantissas, shared_exps, meta = fp32_to_bfp(data, block_size=4, mantissa_bits=8)

        assert mantissas.dtype == np.int8
        assert shared_exps.dtype == np.int8
        assert len(shared_exps) == 1  # 1 block
        assert len(mantissas) == 4

    def test_bfp_roundtrip(self):
        """往返测试"""
        data = np.array([1.0, 2.0, -3.0, 0.5, 1.5, -2.5, 0.0, 4.0], dtype=np.float32)
        mantissas, shared_exps, meta = fp32_to_bfp(data, block_size=4, mantissa_bits=8)

        restored = bfp_to_fp32(mantissas, shared_exps,
                               block_size=4, mantissa_bits=8,
                               original_shape=data.shape)

        # BFP 有精度损失，检查相对误差
        # 8位尾数的量化误差约为 1/128 ≈ 0.8%，但块内共享指数会放大误差
        assert np.allclose(restored, data, rtol=0.15, atol=0.1)

    def test_bfp_multidim(self):
        """多维数组测试"""
        data = np.random.randn(2, 3, 4).astype(np.float32)
        mantissas, shared_exps, meta = fp32_to_bfp(data, block_size=8, mantissa_bits=8)

        restored = bfp_to_fp32(mantissas, shared_exps,
                               block_size=8, mantissa_bits=8,
                               original_shape=data.shape)

        assert restored.shape == data.shape
        # BFP 量化误差取决于块内数值的动态范围
        assert np.allclose(restored, data, rtol=0.15, atol=0.1)

    def test_bfp_block_size(self):
        """不同块大小测试"""
        data = np.random.randn(64).astype(np.float32)

        for block_size in [4, 8, 16, 32]:
            mantissas, shared_exps, meta = fp32_to_bfp(data, block_size=block_size, mantissa_bits=8)
            expected_blocks = (len(data) + block_size - 1) // block_size
            assert len(shared_exps) == expected_blocks


class TestBfpQuantize:
    """通过 quantize 接口测试"""

    def test_bfp16_quantize(self):
        """BFP16 量化"""
        from aidevtools.formats.quantize import quantize

        data = np.random.randn(32).astype(np.float32)
        packed, meta = quantize(data, "bfp16")

        assert meta["format"] == "bfp"
        assert meta["block_size"] == 16
        assert meta["mantissa_bits"] == 8

    def test_bfp8_quantize(self):
        """BFP8 量化"""
        from aidevtools.formats.quantize import quantize

        data = np.random.randn(64).astype(np.float32)
        packed, meta = quantize(data, "bfp8")

        assert meta["format"] == "bfp"
        assert meta["block_size"] == 32
        assert meta["mantissa_bits"] == 4

    def test_list_quantize_includes_bfp(self):
        """检查 BFP 在量化类型列表中"""
        from aidevtools.formats.quantize import list_quantize

        qtypes = list_quantize()
        assert "bfp16" in qtypes
        assert "bfp8" in qtypes


class TestBfpGolden:
    """BFP C++ Golden 实现测试"""

    def test_is_cpp_available(self):
        """检查 C++ 可用性"""
        from aidevtools.formats.custom.bfp.wrapper import is_cpp_available

        result = is_cpp_available()
        assert isinstance(result, bool)

    @pytest.mark.skipif(
        not __import__("aidevtools.formats.custom.bfp.wrapper", fromlist=["is_cpp_available"]).is_cpp_available(),
        reason="BFP C++ 扩展未编译"
    )
    def test_cpp_roundtrip(self):
        """C++ 往返测试"""
        from aidevtools.formats.custom.bfp.wrapper import fp32_to_bfp, bfp_to_fp32

        data = np.random.randn(32).astype(np.float32)
        mantissas, shared_exps = fp32_to_bfp(data, block_size=8, mantissa_bits=8)
        restored = bfp_to_fp32(mantissas, shared_exps, block_size=8, mantissa_bits=8)

        # BFP 量化有精度损失
        assert np.allclose(restored, data, rtol=0.15, atol=0.1)

    @pytest.mark.skipif(
        not __import__("aidevtools.formats.custom.bfp.wrapper", fromlist=["is_cpp_available"]).is_cpp_available(),
        reason="BFP C++ 扩展未编译"
    )
    def test_cpp_python_consistency(self):
        """验证 C++ 和 Python 实现一致性"""
        from aidevtools.formats.custom.bfp.wrapper import fp32_to_bfp as cpp_fp32_to_bfp
        from aidevtools.formats.custom.bfp.golden import fp32_to_bfp as py_fp32_to_bfp

        data = np.array([1.0, 2.0, -3.0, 0.5, 1.5, -2.5, 0.25, 4.0], dtype=np.float32)

        # Python 实现
        py_mantissas, py_exps, _ = py_fp32_to_bfp(data, block_size=4, mantissa_bits=8)

        # C++ 实现
        cpp_mantissas, cpp_exps = cpp_fp32_to_bfp(data, block_size=4, mantissa_bits=8)

        # 结果应完全一致
        np.testing.assert_array_equal(py_mantissas, cpp_mantissas, "尾数不一致")
        np.testing.assert_array_equal(py_exps, cpp_exps, "共享指数不一致")

    @pytest.mark.skipif(
        not __import__("aidevtools.formats.custom.bfp.wrapper", fromlist=["is_cpp_available"]).is_cpp_available(),
        reason="BFP C++ 扩展未编译"
    )
    def test_golden_precision_stats(self):
        """Golden 精度统计 - 验证量化误差在预期范围内"""
        from aidevtools.formats.custom.bfp.wrapper import fp32_to_bfp, bfp_to_fp32

        np.random.seed(42)
        data = np.random.randn(1024).astype(np.float32)

        # 8-bit mantissa 量化
        mantissas, shared_exps = fp32_to_bfp(data, block_size=16, mantissa_bits=8)
        restored = bfp_to_fp32(mantissas, shared_exps, block_size=16, mantissa_bits=8)

        # 计算误差统计 (排除接近零的值，避免相对误差爆炸)
        mask = np.abs(data) > 0.1
        abs_error = np.abs(restored - data)
        rel_error = abs_error[mask] / np.abs(data[mask])

        # 8-bit mantissa 理论误差约 1/128 ≈ 0.78%
        # 由于块内共享指数，实际误差会更大
        assert np.mean(rel_error) < 0.10, f"平均相对误差过大: {np.mean(rel_error):.4f}"
        assert np.percentile(rel_error, 95) < 0.20, f"95%分位误差过大: {np.percentile(rel_error, 95):.4f}"
        assert np.percentile(rel_error, 99) < 0.35, f"99%分位误差过大: {np.percentile(rel_error, 99):.4f}"

    @pytest.mark.skipif(
        not __import__("aidevtools.formats.custom.bfp.wrapper", fromlist=["is_cpp_available"]).is_cpp_available(),
        reason="BFP C++ 扩展未编译"
    )
    def test_golden_edge_cases(self):
        """边界条件测试"""
        from aidevtools.formats.custom.bfp.wrapper import fp32_to_bfp, bfp_to_fp32

        # 测试零值
        zeros = np.zeros(8, dtype=np.float32)
        m, e = fp32_to_bfp(zeros, block_size=4, mantissa_bits=8)
        restored = bfp_to_fp32(m, e, block_size=4, mantissa_bits=8)
        np.testing.assert_array_almost_equal(restored, zeros, decimal=5)

        # 测试同量级小值 (块内动态范围小，精度较好)
        small = np.array([1e-3, 2e-3, 3e-3, 4e-3], dtype=np.float32)
        m, e = fp32_to_bfp(small, block_size=4, mantissa_bits=8)
        restored = bfp_to_fp32(m, e, block_size=4, mantissa_bits=8)
        assert np.allclose(restored, small, rtol=0.05)

        # 测试较大值
        large = np.array([1000.0, 2000.0, -3000.0, 500.0], dtype=np.float32)
        m, e = fp32_to_bfp(large, block_size=4, mantissa_bits=8)
        restored = bfp_to_fp32(m, e, block_size=4, mantissa_bits=8)
        assert np.allclose(restored, large, rtol=0.05)

        # 测试混合量级 (块内共享指数导致小值精度损失)
        mixed = np.array([100.0, 1.0, 0.01, 0.0001], dtype=np.float32)
        m, e = fp32_to_bfp(mixed, block_size=4, mantissa_bits=8)
        restored = bfp_to_fp32(m, e, block_size=4, mantissa_bits=8)
        # 最大值精度好，小值会被"淹没"
        assert np.isclose(restored[0], mixed[0], rtol=0.01)  # 100.0 精度好
        # 小值可能损失严重，只检查不为负
        assert restored[3] >= 0
