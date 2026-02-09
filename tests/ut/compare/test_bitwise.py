"""Bit 级比对分析测试

本测试文件仅测试 BitAnalysisStrategy 的核心功能。
已移除功能（不再测试）：
  - 可视化：print_bit_analysis, print_bit_heatmap, gen_*_svg
  - 模板系统：bit_template, from_template, shared_exponent_bits
  - 统计功能：per_bit_error_count, diff_ratio
  - 模型分析：compare_model_bitwise, ModelBitAnalysis
  - 整数类型：INT8, UINT8
"""

import numpy as np
import pytest

from aidevtools.compare.strategy import (
    FloatFormat,
    BitLayout,
    FP32,
    FP16,
    BFLOAT16,
    BFP16,
    BFP8,
    BFP4,
    BitAnalysisStrategy,
    BitAnalysisResult,
)

from aidevtools.compare.strategy.bit_analysis import WarnLevel


def compare_bitwise(golden, result, fmt=FP32):
    """便捷包装函数"""
    return BitAnalysisStrategy.compare(golden, result, fmt=fmt)


class TestCompareBitwise:
    """compare_bitwise 核心分析"""

    def test_identical_arrays(self):
        """完全相同 → 0 diff, INFO 告警"""
        data = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        r = compare_bitwise(data, data)

        assert r.summary.total_elements == 3
        assert r.summary.diff_elements == 0
        assert r.summary.sign_flip_count == 0
        assert r.summary.exponent_diff_count == 0
        assert r.summary.mantissa_diff_count == 0
        assert not r.has_critical
        assert len(r.warnings) == 1
        assert r.warnings[0].level == WarnLevel.INFO
        assert "Bit-exact" in r.warnings[0].message

    def test_sign_flip(self):
        """符号翻转 → CRITICAL"""
        golden = np.array([1.0, 2.0, -3.0], dtype=np.float32)
        result = np.array([-1.0, 2.0, 3.0], dtype=np.float32)

        r = compare_bitwise(golden, result)

        assert r.summary.sign_flip_count == 2
        assert r.has_critical
        critical = [w for w in r.warnings if w.level == WarnLevel.CRITICAL]
        assert len(critical) >= 1
        assert "Sign flip" in critical[0].message

    def test_large_exponent_diff(self):
        """大指数偏移 → CRITICAL"""
        golden = np.array([1.0, 1.0], dtype=np.float32)
        result = np.array([1024.0, 1.0], dtype=np.float32)  # exponent diff = 10

        r = compare_bitwise(golden, result)

        assert r.summary.exponent_diff_count >= 1
        assert r.has_critical
        critical = [w for w in r.warnings if w.level == WarnLevel.CRITICAL
                    and "Large exponent shift" in w.message]
        assert len(critical) == 1

    def test_small_exponent_diff(self):
        """小指数偏移 → WARNING"""
        golden = np.array([1.0], dtype=np.float32)
        result = np.array([2.0], dtype=np.float32)  # exponent diff = 1

        r = compare_bitwise(golden, result)

        assert r.summary.exponent_diff_count == 1
        warnings = [w for w in r.warnings if w.level == WarnLevel.WARNING]
        assert len(warnings) == 1
        assert "Small exponent shift" in warnings[0].message

    def test_mantissa_only_diff(self):
        """仅尾数差异 → INFO (量化正常)"""
        golden = np.array([1.0], dtype=np.float32)
        # 修改最低位 mantissa
        g_uint = golden.view(np.uint32).copy()
        g_uint[0] ^= 1  # flip LSB
        result = g_uint.view(np.float32).copy()

        r = compare_bitwise(golden, result)

        assert r.summary.diff_elements == 1
        assert r.summary.sign_flip_count == 0
        assert r.summary.exponent_diff_count == 0
        assert r.summary.mantissa_diff_count == 1
        assert not r.has_critical
        info = [w for w in r.warnings if w.level == WarnLevel.INFO
                and "Mantissa-only diff" in w.message]
        assert len(info) == 1

    def test_float16_format(self):
        """float16 格式"""
        golden = np.array([1.0, 2.0, 3.0], dtype=np.float16)
        result = np.array([-1.0, 2.0, 3.0], dtype=np.float16)

        r = compare_bitwise(golden, result, fmt=FloatFormat.FLOAT16)

        assert r.fmt == FloatFormat.FLOAT16
        assert r.summary.sign_flip_count == 1
        assert r.summary.total_elements == 3

    def test_bfloat16_from_float32(self):
        """bfloat16: 从 float32 截取高 16 bits"""
        golden = np.array([1.0, 2.0], dtype=np.float32)
        result = np.array([-1.0, 2.0], dtype=np.float32)

        r = compare_bitwise(golden, result, fmt=FloatFormat.BFLOAT16)

        assert r.fmt == FloatFormat.BFLOAT16
        assert r.summary.sign_flip_count == 1

    def test_shape_mismatch(self):
        """shape 不匹配 → ValueError"""
        golden = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        result = np.array([1.0, 2.0], dtype=np.float32)

        with pytest.raises(ValueError, match="Shape mismatch"):
            compare_bitwise(golden, result)

    def test_warning_indices(self):
        """告警中包含出错索引"""
        golden = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
        result = np.array([-1.0, 2.0, -3.0, 4.0, 5.0], dtype=np.float32)

        r = compare_bitwise(golden, result)

        sign_warnings = [w for w in r.warnings if "Sign flip" in w.message]
        assert len(sign_warnings) == 1
        assert 0 in sign_warnings[0].indices
        assert 2 in sign_warnings[0].indices

    def test_large_array(self):
        """大数组性能 (不超时即通过)"""
        rng = np.random.RandomState(42)
        golden = rng.randn(10000).astype(np.float32)
        result = golden + rng.randn(10000).astype(np.float32) * 0.001

        r = compare_bitwise(golden, result)

        assert r.summary.total_elements == 10000
        assert r.summary.diff_elements > 0


class TestBitLayout:
    """BitLayout 基础功能测试"""

    def test_fp32_preset(self):
        """FP32 预设: 1 sign + 8 exponent + 23 mantissa"""
        assert FP32.sign_bits == 1
        assert FP32.exponent_bits == 8
        assert FP32.mantissa_bits == 23
        assert FP32.total_bits == 32
        assert FP32.name == "fp32"

    def test_fp16_preset(self):
        """FP16 预设: 1 sign + 5 exponent + 10 mantissa"""
        assert FP16.sign_bits == 1
        assert FP16.exponent_bits == 5
        assert FP16.mantissa_bits == 10
        assert FP16.total_bits == 16
        assert FP16.name == "fp16"

    def test_bfp8_preset(self):
        """BFP8 预设: 1 sign + 0 exponent + 7 mantissa"""
        assert BFP8.sign_bits == 1
        assert BFP8.exponent_bits == 0
        assert BFP8.mantissa_bits == 7
        assert BFP8.total_bits == 8
        assert BFP8.name == "bfp8"
        assert BFP8.as_tuple() == (1, 0, 7)

    def test_bfp16_preset(self):
        """BFP16 预设: 1 sign + 0 exponent + 15 mantissa"""
        assert BFP16.sign_bits == 1
        assert BFP16.exponent_bits == 0
        assert BFP16.mantissa_bits == 15
        assert BFP16.total_bits == 16
        assert BFP16.name == "bfp16"
        assert BFP16.as_tuple() == (1, 0, 15)

    def test_bfp4_preset(self):
        """BFP4 预设"""
        assert BFP4.total_bits == 4
        assert BFP4.name == "bfp4"

    def test_custom_layout(self):
        """自定义 BitLayout"""
        fp8 = BitLayout(sign_bits=1, exponent_bits=5, mantissa_bits=2, name="fp8_e5m2")
        assert fp8.total_bits == 8
        assert fp8.name == "fp8_e5m2"
        assert fp8.as_tuple() == (1, 5, 2)

    def test_layout_auto_name(self):
        """无 name 时为空字符串"""
        layout = BitLayout(sign_bits=1, exponent_bits=4, mantissa_bits=3)
        assert layout.name == ""


class TestBitLayoutCompare:
    """使用 BitLayout 的 compare_bitwise"""

    def test_bfp8_identical_uint8(self):
        """BFP8 uint8 数据 — 完全相同"""
        data = np.array([0x3F, 0x7E, 0x01, 0xFF], dtype=np.uint8)
        r = compare_bitwise(data, data, fmt=BFP8)

        assert r.summary.total_elements == 4
        assert r.summary.diff_elements == 0
        assert not r.has_critical
        assert r.fmt.name == "bfp8"

    def test_bfp8_sign_flip(self):
        """BFP8 uint8 — 符号位翻转 (bit 7)"""
        golden = np.array([0b00111111, 0b01010101], dtype=np.uint8)
        result = np.array([0b10111111, 0b01010101], dtype=np.uint8)

        r = compare_bitwise(golden, result, fmt=BFP8)

        assert r.summary.sign_flip_count == 1
        assert r.has_critical

    def test_bfp8_mantissa_only(self):
        """BFP8 uint8 — 仅尾数差异"""
        golden = np.array([0b00111111], dtype=np.uint8)
        result = np.array([0b00111110], dtype=np.uint8)  # LSB flip

        r = compare_bitwise(golden, result, fmt=BFP8)

        assert r.summary.diff_elements == 1
        assert r.summary.sign_flip_count == 0
        assert r.summary.mantissa_diff_count == 1
        assert not r.has_critical

    def test_custom_fp8_e5m2(self):
        """自定义 FP8 E5M2"""
        layout = BitLayout(sign_bits=1, exponent_bits=5, mantissa_bits=2, name="fp8_e5m2")
        golden = np.array([0b01111100, 0b00111100], dtype=np.uint8)
        result = np.array([0b01111100, 0b10111100], dtype=np.uint8)

        r = compare_bitwise(golden, result, fmt=layout)

        assert r.summary.total_elements == 2
        assert r.summary.sign_flip_count == 1
        assert r.fmt.name == "fp8_e5m2"

    def test_bfp8_from_float32(self):
        """BFP8 格式但输入是 float32 — 取低 8 bit 的 uint32 表示"""
        golden = np.array([1.0, 2.0], dtype=np.float32)
        result = np.array([-1.0, 2.0], dtype=np.float32)
        # 对 float32 用 BFP8 会走 view(uint32) 路径
        r = compare_bitwise(golden, result, fmt=BFP8)
        assert r.summary.total_elements == 2
