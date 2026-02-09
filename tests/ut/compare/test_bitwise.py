"""Bit 级比对分析测试"""

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

# WarnLevel 没有在 __init__.py 中导出，直接从模块导入
from aidevtools.compare.strategy.bit_analysis import WarnLevel

# 旧 API 兼容层 - 将 compare_bitwise 映射到新的静态方法
def compare_bitwise(golden, result, fmt=FP32):
    """兼容旧 API - 自动推断格式"""
    return BitAnalysisStrategy.compare(golden, result, fmt=fmt)

# 删除已移除的类型/函数（测试中如果用到需要更新）
# INT8, UINT8, BitDiffSummary, BitWarning, ModelBitAnalysis
# compare_model_bitwise, print_bit_template, print_bit_analysis
# print_bit_heatmap, print_model_bit_analysis
# gen_bit_heatmap_svg, gen_perbit_bar_svg


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
        assert "符号位翻转" in critical[0].message

    def test_large_exponent_diff(self):
        """大指数偏移 → CRITICAL"""
        golden = np.array([1.0, 1.0], dtype=np.float32)
        result = np.array([1024.0, 1.0], dtype=np.float32)  # exponent diff = 10

        r = compare_bitwise(golden, result)

        assert r.summary.exponent_diff_count >= 1
        assert r.has_critical
        critical = [w for w in r.warnings if w.level == WarnLevel.CRITICAL
                    and "指数域大偏移" in w.message]
        assert len(critical) == 1

    def test_small_exponent_diff(self):
        """小指数偏移 → WARNING"""
        golden = np.array([1.0], dtype=np.float32)
        result = np.array([2.0], dtype=np.float32)  # exponent diff = 1

        r = compare_bitwise(golden, result)

        assert r.summary.exponent_diff_count == 1
        warnings = [w for w in r.warnings if w.level == WarnLevel.WARNING]
        assert len(warnings) == 1
        assert "指数域偏移 (±1)" in warnings[0].message

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
                and "仅尾数差异" in w.message]
        assert len(info) == 1

    def test_per_bit_error_count(self):
        """per-bit 错误计数"""
        golden = np.zeros(100, dtype=np.float32)
        result = golden.copy()
        # flip bit 0 (LSB mantissa) for all elements
        r_uint = result.view(np.uint32)
        r_uint[:] ^= 1
        result = r_uint.view(np.float32)

        r = compare_bitwise(golden.view(np.float32), result)

        assert r.summary.per_bit_error_count[0] == 100  # bit 0 has 100 errors
        assert r.summary.per_bit_error_count[1] == 0    # bit 1 has 0 errors

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

        r = compare_bitwise(golden, result, max_warning_indices=5)

        sign_warnings = [w for w in r.warnings if "符号位翻转" in w.message]
        assert len(sign_warnings) == 1
        assert 0 in sign_warnings[0].indices
        assert 2 in sign_warnings[0].indices

    def test_diff_ratio(self):
        """diff_ratio 计算"""
        golden = np.ones(100, dtype=np.float32)
        result = golden.copy()
        result[:25] = -1.0  # 25% diff

        r = compare_bitwise(golden, result)

        assert r.summary.diff_ratio == 0.25

    def test_large_array(self):
        """大数组性能 (不超时即通过)"""
        rng = np.random.RandomState(42)
        golden = rng.randn(10000).astype(np.float32)
        result = golden + rng.randn(10000).astype(np.float32) * 0.001

        r = compare_bitwise(golden, result)

        assert r.summary.total_elements == 10000
        assert r.summary.diff_elements > 0


class TestPrintBitAnalysis:
    """print_bit_analysis 输出格式"""

    def test_output(self, capsys):
        """验证输出包含关键信息"""
        golden = np.array([1.0, -2.0], dtype=np.float32)
        result = np.array([-1.0, -2.0], dtype=np.float32)

        r = compare_bitwise(golden, result)
        print_bit_analysis(r, name="test_op")

        captured = capsys.readouterr()
        assert "[test_op]" in captured.out
        assert "Bit-Level Analysis" in captured.out
        assert "float32" in captured.out
        assert "Sign flips:" in captured.out
        assert "[!]" in captured.out  # CRITICAL mark


class TestPrintBitHeatmap:
    """print_bit_heatmap 输出"""

    def test_heatmap_exact(self, capsys):
        """完全相同 → 全 '.'"""
        data = np.ones(1024, dtype=np.float32)
        print_bit_heatmap(data, data, block_size=256, cols=10)

        captured = capsys.readouterr()
        assert "Bit Heatmap" in captured.out
        assert "...." in captured.out  # 4 blocks, all exact

    def test_heatmap_mixed(self, capsys):
        """有差异"""
        golden = np.ones(512, dtype=np.float32)
        result = golden.copy()
        result[256:] = -1.0  # second half sign-flipped

        print_bit_heatmap(golden, result, block_size=256, cols=10)

        captured = capsys.readouterr()
        assert "." in captured.out
        assert "#" in captured.out  # heavy diff block


class TestGenBitHeatmapSvg:
    """SVG 热力图生成"""

    def test_generate_svg(self, tmp_path):
        """生成 SVG 文件"""
        golden = np.ones(1024, dtype=np.float32)
        result = golden.copy()
        result[512:] += 0.001

        svg_path = str(tmp_path / "bit_heatmap.svg")
        gen_bit_heatmap_svg(golden, result, svg_path, block_size=256)

        content = (tmp_path / "bit_heatmap.svg").read_text()
        assert "<svg" in content
        assert "</svg>" in content
        assert "rect" in content


class TestGenPerbitBarSvg:
    """per-bit 条形图 SVG"""

    def test_generate_bar_svg(self, tmp_path):
        """生成 per-bit 条形图"""
        golden = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        result = np.array([-1.0, 2.0, 3.5], dtype=np.float32)

        r = compare_bitwise(golden, result)
        svg_path = str(tmp_path / "perbit.svg")
        gen_perbit_bar_svg(r, svg_path)

        content = (tmp_path / "perbit.svg").read_text()
        assert "<svg" in content
        assert "Per-Bit Error Distribution" in content
        assert "sign" in content
        assert "exponent" in content
        assert "mantissa" in content
        # 应有 32 根柱子 (float32)
        assert content.count('title="bit') == 32


class TestBitLayout:
    """BitLayout 自定义格式测试"""

    def test_fp32_preset(self):
        """FP32 预设: 1 sign + 8 exponent + 23 mantissa"""
        assert FP32.sign_bits == 1
        assert FP32.exponent_bits == 8
        assert FP32.mantissa_bits == 23
        assert FP32.total_bits == 32
        assert FP32.shared_exponent_bits == 0
        assert FP32.block_size == 1
        assert FP32.display_name == "fp32"
        assert FP32.bit_template == "SEEEEEEEEMMMMMMMMMMMMMMMMMMMMMMM"

    def test_fp16_preset(self):
        """FP16 预设: 1 sign + 5 exponent + 10 mantissa"""
        assert FP16.sign_bits == 1
        assert FP16.exponent_bits == 5
        assert FP16.mantissa_bits == 10
        assert FP16.total_bits == 16
        assert FP16.shared_exponent_bits == 0
        assert FP16.display_name == "fp16"
        assert FP16.bit_template == "SEEEEEMMMMMMMMMM"

    def test_bfp8_preset(self):
        """BFP8 预设: 1 sign + 0 exponent + 7 mantissa"""
        assert BFP8.sign_bits == 1
        assert BFP8.exponent_bits == 0
        assert BFP8.mantissa_bits == 7
        assert BFP8.total_bits == 8
        assert BFP8.shared_exponent_bits == 8
        assert BFP8.block_size == 16
        assert BFP8.display_name == "bfp8"
        assert BFP8.as_tuple() == (1, 0, 7)

    def test_bfp16_preset(self):
        """BFP16 预设: 1 sign + 0 exponent + 15 mantissa"""
        assert BFP16.sign_bits == 1
        assert BFP16.exponent_bits == 0
        assert BFP16.mantissa_bits == 15
        assert BFP16.total_bits == 16
        assert BFP16.shared_exponent_bits == 8
        assert BFP16.block_size == 16
        assert BFP16.display_name == "bfp16"
        assert BFP16.as_tuple() == (1, 0, 15)
        assert BFP16.bit_template == "SMMMMMMMMMMMMMMM"

    def test_bfp4_preset(self):
        """BFP4 预设"""
        assert BFP4.total_bits == 4
        assert BFP4.display_name == "bfp4"

    def test_int8_preset(self):
        """INT8 预设: 1 sign + 7 data"""
        assert INT8.sign_bits == 1
        assert INT8.exponent_bits == 0
        assert INT8.mantissa_bits == 7
        assert INT8.display_name == "int8"

    def test_uint8_preset(self):
        """UINT8 预设: 0 sign + 8 data"""
        assert UINT8.sign_bits == 0
        assert UINT8.exponent_bits == 0
        assert UINT8.mantissa_bits == 8
        assert UINT8.display_name == "uint8"

    def test_custom_layout(self):
        """自定义 BitLayout"""
        fp8 = BitLayout(sign_bits=1, exponent_bits=5, mantissa_bits=2, name="fp8_e5m2")
        assert fp8.total_bits == 8
        assert fp8.display_name == "fp8_e5m2"
        assert fp8.as_tuple() == (1, 5, 2)

    def test_layout_auto_name(self):
        """无 name 时自动生成显示名"""
        layout = BitLayout(sign_bits=1, exponent_bits=4, mantissa_bits=3)
        assert "s1" in layout.display_name
        assert "e4" in layout.display_name
        assert "m3" in layout.display_name

    def test_layout_shared_exp_name(self):
        """共享指数格式的显示名"""
        layout = BitLayout(sign_bits=1, exponent_bits=0, mantissa_bits=7,
                           shared_exponent_bits=8, block_size=16)
        assert "shared_exp" in layout.display_name

    # --- bit template ---

    def test_float32_template(self):
        """float32 bit 模板"""
        layout = BitLayout(sign_bits=1, exponent_bits=8, mantissa_bits=23, name="float32")
        assert layout.bit_template == "S" + "E" * 8 + "M" * 23
        assert len(layout.bit_template) == 32

    def test_bfp8_template(self):
        """BFP8 bit 模板: S + 7M"""
        assert BFP8.bit_template == "SMMMMMMM"

    def test_bfp4_template(self):
        """BFP4 bit 模板: S + 3M"""
        assert BFP4.bit_template == "SMMM"

    def test_uint8_template(self):
        """UINT8 无符号: 纯 I (integer)"""
        assert UINT8.bit_template == "IIIIIIII"

    def test_fp8_e5m2_template(self):
        """FP8 E5M2 模板"""
        fp8 = BitLayout(sign_bits=1, exponent_bits=5, mantissa_bits=2)
        assert fp8.bit_template == "SEEEEE" + "MM"

    def test_bit_template_spaced(self):
        """带空格分隔的 bit 模板"""
        layout = BitLayout(sign_bits=1, exponent_bits=8, mantissa_bits=23)
        assert layout.bit_template_spaced == "S EEEEEEEE MMMMMMMMMMMMMMMMMMMMMMM"

    def test_bfp8_template_spaced(self):
        """BFP8 带空格模板 (无 exponent)"""
        assert BFP8.bit_template_spaced == "S MMMMMMM"

    def test_uint8_template_spaced(self):
        """UINT8 带空格模板 (无 sign 无 exp)"""
        assert UINT8.bit_template_spaced == "IIIIIIII"


class TestBitLayoutFromTemplate:
    """from_template 构造"""

    def test_from_template_fp8_e4m3(self):
        """FP8 E4M3 via template"""
        layout = BitLayout.from_template(
            "SEEEEMMMM",
            name="fp8_e4m3",
        )
        assert layout.total_bits == 9
        assert layout.sign_bits == 1
        assert layout.exponent_bits == 4
        assert layout.mantissa_bits == 4
        assert layout.bit_template == "SEEEEMMMM"

    def test_from_template_custom_with_parity(self):
        """自定义格式: flag + exp + data + parity"""
        layout = BitLayout.from_template(
            "FFEEEDDDDP",
            name="custom10",
        )
        assert layout.total_bits == 10
        assert layout.sign_bits == 0  # no "sign" letter → 0
        assert layout.exponent_bits == 3
        assert layout.mantissa_bits == 7  # flag(2) + data(4) + parity(1) all counted as "mantissa"
        assert layout.bit_template == "FFEEEDDDDP"
        assert layout.bit_template_spaced == "FF EEE DDDD P"

    def test_from_template_bfp8_shared(self):
        """BFP8 via template — 含共享指数"""
        layout = BitLayout.from_template(
            "SMMMMMMM",
            name="bfp8_tmpl",
            shared_template="EEEEEEEE",
            block_size=16,
        )
        assert layout.total_bits == 8
        assert layout.sign_bits == 1
        assert layout.exponent_bits == 0
        assert layout.mantissa_bits == 7
        assert layout.shared_exponent_bits == 8
        assert layout.block_size == 16
        assert layout.bit_template == "SMMMMMMM"

    def test_from_template_int8(self):
        """INT8 via template"""
        layout = BitLayout.from_template(
            "SIIIIIII",
            name="int8_tmpl",
        )
        assert layout.sign_bits == 1
        assert layout.bit_template == "SIIIIIII"
        assert layout.bit_group_labels == {'S': 'sign', 'I': 'integer'}

    def test_unified_template_bfp8(self):
        """BFP8 统一模板语法: EEEEEEEE(SMMMMMMM)*16"""
        layout = BitLayout(template="EEEEEEEE(SMMMMMMM)*16", name="bfp8_u")
        assert layout.sign_bits == 1
        assert layout.exponent_bits == 0
        assert layout.mantissa_bits == 7
        assert layout.total_bits == 8
        assert layout.shared_exponent_bits == 8
        assert layout.block_size == 16
        assert layout.bit_template == "SMMMMMMM"
        assert layout.shared_template == "EEEEEEEE"

    def test_unified_template_bfp16(self):
        """BFP16 统一模板语法"""
        layout = BitLayout(template="EEEEEEEE(SMMMMMMMMMMMMMMM)*16", name="bfp16_u")
        assert layout.total_bits == 16
        assert layout.sign_bits == 1
        assert layout.mantissa_bits == 15
        assert layout.shared_exponent_bits == 8
        assert layout.block_size == 16

    def test_unified_template_plain(self):
        """普通模板 (无括号): 全部是 per-element"""
        layout = BitLayout(template="SEEEEMMMM", name="fp8_e4m3")
        assert layout.sign_bits == 1
        assert layout.exponent_bits == 4
        assert layout.mantissa_bits == 4
        assert layout.total_bits == 9
        assert layout.shared_exponent_bits == 0
        assert layout.block_size == 1
        assert layout.shared_template == ""


class TestPrintBitTemplate:
    """print_bit_template 输出"""

    def test_float32_template(self, capsys):
        """float32 模板输出"""
        print_bit_template(FloatFormat.FLOAT32)
        captured = capsys.readouterr()
        assert "float32" in captured.out
        assert "32 bits" in captured.out
        assert "S EEEEEEEE" in captured.out
        assert "S=sign" in captured.out
        assert "E=exponent" in captured.out
        assert "M=mantissa" in captured.out

    def test_bfp8_template(self, capsys):
        """BFP8 模板输出 — 含共享指数"""
        print_bit_template(BFP8)
        captured = capsys.readouterr()
        assert "bfp8" in captured.out
        assert "S MMMMMMM" in captured.out
        assert "Shared" in captured.out
        assert "EEEEEEEE" in captured.out
        assert "block of 16" in captured.out
        assert "shared_exponent" in captured.out

    def test_custom_layout(self, capsys):
        """自定义格式模板输出"""
        layout = BitLayout(sign_bits=1, exponent_bits=5, mantissa_bits=2, name="fp8_e5m2")
        print_bit_template(layout)
        captured = capsys.readouterr()
        assert "fp8_e5m2" in captured.out
        assert "S EEEEE MM" in captured.out

    def test_analysis_includes_template(self, capsys):
        """print_bit_analysis 输出含 bit layout 模板"""
        golden = np.array([0x3F, 0x7E], dtype=np.uint8)
        result = np.array([0xBF, 0x7E], dtype=np.uint8)
        r = compare_bitwise(golden, result, fmt=BFP8)
        print_bit_analysis(r, name="test")

        captured = capsys.readouterr()
        assert "Bit layout:" in captured.out
        assert "S MMMMMMM" in captured.out
        assert "S=sign" in captured.out


class TestBitLayoutCompare:
    """使用 BitLayout 的 compare_bitwise"""

    def test_bfp8_identical_uint8(self):
        """BFP8 uint8 数据 — 完全相同"""
        data = np.array([0x3F, 0x7E, 0x01, 0xFF], dtype=np.uint8)
        r = compare_bitwise(data, data, fmt=BFP8)

        assert r.summary.total_elements == 4
        assert r.summary.diff_elements == 0
        assert not r.has_critical
        assert r.format_name == "bfp8"

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

    def test_bfp8_per_bit_count(self):
        """BFP8: 8 个 bit position"""
        data = np.zeros(10, dtype=np.uint8)
        r = compare_bitwise(data, data, fmt=BFP8)
        assert len(r.summary.per_bit_error_count) == 8

    def test_bfp8_print_analysis(self, capsys):
        """BFP8 print_bit_analysis 不 crash"""
        golden = np.array([0x3F, 0x7E], dtype=np.uint8)
        result = np.array([0xBF, 0x7E], dtype=np.uint8)
        r = compare_bitwise(golden, result, fmt=BFP8)
        print_bit_analysis(r, name="bfp8_test")

        captured = capsys.readouterr()
        assert "bfp8" in captured.out
        assert "Bit-Level Analysis" in captured.out

    def test_bfp8_print_heatmap(self, capsys):
        """BFP8 print_bit_heatmap 不 crash"""
        golden = np.zeros(32, dtype=np.uint8)
        result = golden.copy()
        result[0] = 0xFF
        print_bit_heatmap(golden, result, fmt=BFP8, block_size=8, cols=10)

        captured = capsys.readouterr()
        assert "Bit Heatmap" in captured.out

    def test_bfp8_gen_heatmap_svg(self, tmp_path):
        """BFP8 SVG 热力图"""
        golden = np.zeros(64, dtype=np.uint8)
        result = golden.copy()
        result[:8] = 0xFF
        svg_path = str(tmp_path / "bfp8_heatmap.svg")
        gen_bit_heatmap_svg(golden, result, svg_path, fmt=BFP8, block_size=16)

        content = (tmp_path / "bfp8_heatmap.svg").read_text()
        assert "<svg" in content

    def test_bfp8_gen_perbit_bar_svg(self, tmp_path):
        """BFP8 per-bit 条形图"""
        golden = np.array([0x3F, 0x7E, 0x01], dtype=np.uint8)
        result = np.array([0xBF, 0x7E, 0x81], dtype=np.uint8)
        r = compare_bitwise(golden, result, fmt=BFP8)

        svg_path = str(tmp_path / "bfp8_perbit.svg")
        gen_perbit_bar_svg(r, svg_path)

        content = (tmp_path / "bfp8_perbit.svg").read_text()
        assert "<svg" in content
        assert "Per-Bit Error Distribution" in content
        # BFP8: 8 个 bit position
        assert content.count('title="bit') == 8

    def test_custom_fp8_e5m2(self):
        """自定义 FP8 E5M2"""
        layout = BitLayout(sign_bits=1, exponent_bits=5, mantissa_bits=2, name="fp8_e5m2")
        golden = np.array([0b01111100, 0b00111100], dtype=np.uint8)
        result = np.array([0b01111100, 0b10111100], dtype=np.uint8)

        r = compare_bitwise(golden, result, fmt=layout)

        assert r.summary.total_elements == 2
        assert r.summary.sign_flip_count == 1
        assert r.format_name == "fp8_e5m2"

    def test_bfp8_from_float32(self):
        """BFP8 格式但输入是 float32 — 取低 8 bit 的 uint32 表示"""
        golden = np.array([1.0, 2.0], dtype=np.float32)
        result = np.array([-1.0, 2.0], dtype=np.float32)
        # 对 float32 用 BFP8 会走 view(uint32) 路径
        r = compare_bitwise(golden, result, fmt=BFP8)
        assert r.summary.total_elements == 2


class TestModelBitAnalysis:
    """一键式模型 bit 比对测试"""

    def test_per_op_only(self):
        """仅逐算子比对 (无 final_pair)"""
        g1 = np.array([0x3F, 0x7E, 0x01], dtype=np.uint8)
        d1 = np.array([0xBF, 0x7E, 0x01], dtype=np.uint8)
        g2 = np.array([0x10, 0x20], dtype=np.uint8)
        d2 = np.array([0x10, 0x20], dtype=np.uint8)

        result = compare_model_bitwise(
            per_op_pairs={"op_a": (g1, d1), "op_b": (g2, d2)},
            fmt=BFP8,
        )
        assert isinstance(result, ModelBitAnalysis)
        assert len(result.per_op) == 2
        assert result.global_result is None
        assert result.per_op["op_a"].summary.sign_flip_count == 1
        assert result.per_op["op_b"].summary.diff_elements == 0

    def test_with_final_pair(self):
        """逐算子 + 全局"""
        g1 = np.array([0xFF], dtype=np.uint8)
        d1 = np.array([0x00], dtype=np.uint8)

        g_final = np.array([0x3F, 0x7E], dtype=np.uint8)
        d_final = np.array([0xBF, 0x7E], dtype=np.uint8)

        result = compare_model_bitwise(
            per_op_pairs={"op_a": (g1, d1)},
            fmt=BFP8,
            final_pair=(g_final, d_final),
        )
        assert result.global_result is not None
        assert result.global_result.summary.total_elements == 2
        assert result.global_result.summary.sign_flip_count == 1
        assert result.per_op["op_a"].summary.diff_elements == 1

    def test_has_critical(self):
        """has_critical 聚合"""
        g = np.array([0x3F], dtype=np.uint8)
        d = np.array([0xBF], dtype=np.uint8)  # sign flip

        result = compare_model_bitwise(
            per_op_pairs={"sign_flip_op": (g, d)},
            fmt=BFP8,
        )
        assert result.has_critical

    def test_no_critical(self):
        """无 CRITICAL"""
        g = np.array([0x3F], dtype=np.uint8)
        d = np.array([0x3E], dtype=np.uint8)  # LSB mantissa diff only

        result = compare_model_bitwise(
            per_op_pairs={"mant_op": (g, d)},
            fmt=BFP8,
        )
        assert not result.has_critical

    def test_print_model(self, capsys):
        """print_model_bit_analysis 输出包含关键信息"""
        g = np.array([0x3F, 0x7E], dtype=np.uint8)
        d = np.array([0xBF, 0x7E], dtype=np.uint8)

        result = compare_model_bitwise(
            per_op_pairs={"test_op": (g, d)},
            fmt=BFP8,
            final_pair=(g, d),
        )
        print_model_bit_analysis(result, name="TestModel")

        captured = capsys.readouterr()
        assert "[TestModel" in captured.out
        assert "逐算子" in captured.out
        assert "test_op" in captured.out
        assert "合计" in captured.out
        assert "Global" in captured.out
