"""Compare diff 子命令系统测试

双路径架构验证:
- exact / bit_analysis: 源格式字节级比对
- fuzzy / sanity / blocked: 反量化 fp32 比对
- 文件名自动解读: {op}_{qtype}_{NxMxK}[_result].{ext}
"""
import numpy as np
import pytest

from aidevtools.commands.compare import cmd_compare
from aidevtools.formats.base import save
from aidevtools.formats.quantize import quantize


def _make_hex_pair(tmp_path, golden_fp32, result_fp32, qtype="bfpp8",
                   g_name="golden.txt", r_name="result.txt"):
    """辅助: 将两组 fp32 数据量化为 packed bytes 再存成 hex-text 文件"""
    g_packed, _ = quantize(golden_fp32, qtype)
    r_packed, _ = quantize(result_fp32, qtype)
    g_path = tmp_path / g_name
    r_path = tmp_path / r_name
    save(str(g_path), g_packed, fmt="hex_text")
    save(str(r_path), r_packed, fmt="hex_text")
    return str(g_path), str(r_path)


class TestDiffIdentical:
    """相同数据 → PASS"""

    def test_diff_identical_bfpp8(self, tmp_path):
        data = np.random.randn(64).astype(np.float32) * 0.5
        g_path, r_path = _make_hex_pair(tmp_path, data, data, qtype="bfpp8")
        ret = cmd_compare(
            action="diff",
            golden=g_path,
            result=r_path,
            format="hex_text",
            qtype="bfpp8",
            shape="64",
        )
        assert ret == 0

    def test_diff_identical_gfloat8(self, tmp_path):
        data = np.array([1.0, -0.5, 0.25, 3.0], dtype=np.float32)
        g_path, r_path = _make_hex_pair(tmp_path, data, data, qtype="gfloat8")
        ret = cmd_compare(
            action="diff",
            golden=g_path,
            result=r_path,
            format="hex_text",
            qtype="gfloat8",
            shape="4",
        )
        assert ret == 0


class TestDiffDifferent:
    """明显不同的数据 → FAIL"""

    def test_diff_different_bfpp8(self, tmp_path):
        g = np.ones(64, dtype=np.float32)
        r = g + 10.0
        g_path, r_path = _make_hex_pair(tmp_path, g, r, qtype="bfpp8")
        ret = cmd_compare(
            action="diff",
            golden=g_path,
            result=r_path,
            format="hex_text",
            qtype="bfpp8",
            shape="64",
        )
        assert ret == 1


class TestDiffMissingArgs:
    """缺少参数 → return 1"""

    def test_diff_missing_golden(self):
        ret = cmd_compare(action="diff", golden="", result="")
        assert ret == 1

    def test_diff_missing_result(self, tmp_path):
        p = tmp_path / "dummy.txt"
        p.write_text("00\n")
        ret = cmd_compare(action="diff", golden=str(p), result="")
        assert ret == 1


class TestDiffReportOutput:
    """验证报告输出包含关键字"""

    def test_diff_report_output(self, tmp_path, capsys):
        data = np.random.randn(64).astype(np.float32) * 0.5
        g_path, r_path = _make_hex_pair(tmp_path, data, data, qtype="bfpp8")
        cmd_compare(
            action="diff",
            golden=g_path,
            result=r_path,
            format="hex_text",
            qtype="bfpp8",
            shape="64",
        )
        captured = capsys.readouterr()
        assert "exact" in captured.out
        assert "cosine" in captured.out
        assert "PASS" in captured.out


class TestDiffExactOnRawBytes:
    """验证 exact 比的是源格式字节"""

    def test_exact_pass_same_bytes(self, tmp_path, capsys):
        """相同 hex 文件 → exact=Y"""
        data = np.random.randn(64).astype(np.float32) * 0.5
        g_path, r_path = _make_hex_pair(tmp_path, data, data, qtype="bfpp8")
        cmd_compare(
            action="diff",
            golden=g_path,
            result=r_path,
            format="hex_text",
            qtype="bfpp8",
            shape="64",
        )
        captured = capsys.readouterr()
        lines = captured.out.split("\n")
        data_line = [l for l in lines if "golden.txt" in l and "Bit Analysis" not in l]
        assert len(data_line) == 1
        assert "Y" in data_line[0].split()[1]  # exact 列

    def test_exact_fail_different_bytes(self, tmp_path, capsys):
        """不同 hex 文件 → exact=N"""
        g = np.ones(64, dtype=np.float32) * 0.5
        r = g + 0.2
        g_path, r_path = _make_hex_pair(tmp_path, g, r, qtype="bfpp8")
        cmd_compare(
            action="diff",
            golden=g_path,
            result=r_path,
            format="hex_text",
            qtype="bfpp8",
            shape="64",
        )
        captured = capsys.readouterr()
        lines = captured.out.split("\n")
        data_line = [l for l in lines if "golden.txt" in l and "Bit Analysis" not in l]
        assert len(data_line) == 1
        assert "N" in data_line[0].split()[1]  # exact 列 = N


class TestDiffBitAnalysis:
    """验证 bit_analysis 输出"""

    def test_bit_analysis_bfpp8_identical(self, tmp_path, capsys):
        data = np.random.randn(64).astype(np.float32) * 0.5
        g_path, r_path = _make_hex_pair(tmp_path, data, data, qtype="bfpp8")
        cmd_compare(
            action="diff", golden=g_path, result=r_path,
            format="hex_text", qtype="bfpp8", shape="64",
        )
        captured = capsys.readouterr()
        assert "Bit Analysis" in captured.out
        assert "Bit-exact match" in captured.out

    def test_bit_analysis_bfpp8_different(self, tmp_path, capsys):
        g = np.ones(64, dtype=np.float32)
        r = g + 10.0
        g_path, r_path = _make_hex_pair(tmp_path, g, r, qtype="bfpp8")
        cmd_compare(
            action="diff", golden=g_path, result=r_path,
            format="hex_text", qtype="bfpp8", shape="64",
        )
        captured = capsys.readouterr()
        assert "Bit Analysis" in captured.out
        assert "Diff elements" in captured.out

    def test_bit_analysis_gfloat8(self, tmp_path, capsys):
        g = np.array([1.0, -0.5, 0.25, 3.0], dtype=np.float32)
        g_path, r_path = _make_hex_pair(tmp_path, g, g, qtype="gfloat8")
        cmd_compare(
            action="diff", golden=g_path, result=r_path,
            format="hex_text", qtype="gfloat8", shape="4",
        )
        captured = capsys.readouterr()
        assert "Bit Analysis" in captured.out


class TestDiffBlocked:
    """验证 BFP 类型输出 blocked heatmap"""

    def test_blocked_heatmap_bfpp8(self, tmp_path, capsys):
        data = np.random.randn(64).astype(np.float32) * 0.5
        g_path, r_path = _make_hex_pair(tmp_path, data, data, qtype="bfpp8")
        cmd_compare(
            action="diff", golden=g_path, result=r_path,
            format="hex_text", qtype="bfpp8", shape="64",
        )
        captured = capsys.readouterr()
        assert "Block Heatmap" in captured.out

    def test_no_blocked_for_gfloat(self, tmp_path, capsys):
        data = np.array([1.0, -0.5, 0.25, 3.0], dtype=np.float32)
        g_path, r_path = _make_hex_pair(tmp_path, data, data, qtype="gfloat8")
        cmd_compare(
            action="diff", golden=g_path, result=r_path,
            format="hex_text", qtype="gfloat8", shape="4",
        )
        captured = capsys.readouterr()
        assert "Block Heatmap" not in captured.out


class TestDiffEngine:
    """不同引擎选择"""

    def test_diff_quick_engine(self, tmp_path):
        data = np.random.randn(64).astype(np.float32) * 0.5
        g_path, r_path = _make_hex_pair(tmp_path, data, data, qtype="bfpp8")
        ret = cmd_compare(
            action="diff", golden=g_path, result=r_path,
            format="hex_text", qtype="bfpp8", shape="64", engine="quick",
        )
        assert ret == 0


class TestDiffAutoDetect:
    """文件名自动解读测试"""

    def test_auto_detect_bfpp8(self, tmp_path):
        """从文件名自动解读 qtype + shape + fmt，无需手动指定"""
        data = np.random.randn(64).astype(np.float32) * 0.5
        g_path, r_path = _make_hex_pair(
            tmp_path, data, data, qtype="bfpp8",
            g_name="softmax_bfpp8_64.txt",
            r_name="softmax_bfpp8_64_result.txt",
        )
        # 不传 format / qtype / shape，全部自动解读
        ret = cmd_compare(action="diff", golden=g_path, result=r_path)
        assert ret == 0

    def test_auto_detect_multidim(self, tmp_path):
        """多维 shape 自动解读"""
        data = np.random.randn(2, 16, 64).astype(np.float32) * 0.5
        g_path, r_path = _make_hex_pair(
            tmp_path, data, data, qtype="bfpp8",
            g_name="linear_0_bfpp8_2x16x64.txt",
            r_name="linear_0_bfpp8_2x16x64_result.txt",
        )
        ret = cmd_compare(action="diff", golden=g_path, result=r_path)
        assert ret == 0

    def test_auto_detect_gfloat8(self, tmp_path):
        """gfloat8 自动解读"""
        data = np.array([1.0, -0.5, 0.25, 3.0], dtype=np.float32)
        g_path, r_path = _make_hex_pair(
            tmp_path, data, data, qtype="gfloat8",
            g_name="relu_gfloat8_4.txt",
            r_name="relu_gfloat8_4_result.txt",
        )
        ret = cmd_compare(action="diff", golden=g_path, result=r_path)
        assert ret == 0

    def test_auto_detect_op_name_in_report(self, tmp_path, capsys):
        """报告中使用算子名"""
        data = np.random.randn(64).astype(np.float32) * 0.5
        g_path, r_path = _make_hex_pair(
            tmp_path, data, data, qtype="bfpp8",
            g_name="softmax_bfpp8_64.txt",
            r_name="softmax_bfpp8_64_result.txt",
        )
        cmd_compare(action="diff", golden=g_path, result=r_path)
        captured = capsys.readouterr()
        assert "softmax" in captured.out

    def test_manual_override(self, tmp_path):
        """手动指定 qtype/shape 优先于文件名解读"""
        data = np.random.randn(64).astype(np.float32) * 0.5
        g_path, r_path = _make_hex_pair(
            tmp_path, data, data, qtype="bfpp8",
            g_name="softmax_bfpp8_64.txt",
            r_name="softmax_bfpp8_64_result.txt",
        )
        # 手动指定覆盖文件名
        ret = cmd_compare(
            action="diff", golden=g_path, result=r_path,
            format="hex_text", qtype="bfpp8", shape="64",
        )
        assert ret == 0


class TestDiffRawFormat:
    """diff action 也可以用 raw 格式"""

    def test_diff_raw_float32(self, tmp_path):
        data = np.random.randn(32).astype(np.float32)
        g_path = str(tmp_path / "g.bin")
        r_path = str(tmp_path / "r.bin")
        data.tofile(g_path)
        data.tofile(r_path)
        ret = cmd_compare(
            action="diff", golden=g_path, result=r_path,
            format="raw", qtype="float32", shape="32",
        )
        assert ret == 0


def _make_packed_pair(tmp_path, packed_g, packed_r,
                      g_name="golden.txt", r_name="result.txt"):
    """辅助: 直接存 packed bytes 为 hex-text (跳过量化，精确控制字节内容)"""
    g_path = tmp_path / g_name
    r_path = tmp_path / r_name
    save(str(g_path), packed_g, fmt="hex_text")
    save(str(r_path), packed_r, fmt="hex_text")
    return str(g_path), str(r_path)


class TestThreeLevelCompare:
    """三级系统比数: exact + bit analysis + blocked heatmap

    三个梯度:
    1. 完全一样 → 全 PASS
    2. 1 block 1 bit 差异 → exact=N, bit analysis 定位, blocked 局部异常
    3. 一半不同 → 全面 FAIL
    """

    def test_identical_full_pass(self, tmp_path, capsys):
        """完全一样 → exact=Y, Bit-exact match, Block Heatmap 全绿"""
        data = np.array([0.5, -0.3, 0.1, 0.7] * 16, dtype=np.float32)
        packed, _ = quantize(data, "bfpp8")
        g_path, r_path = _make_packed_pair(tmp_path, packed, packed)
        ret = cmd_compare(
            action="diff", golden=g_path, result=r_path,
            format="hex_text", qtype="bfpp8", shape="64",
        )
        out = capsys.readouterr().out

        assert ret == 0
        # Level 1: exact — 字节精确匹配
        data_line = [l for l in out.split("\n") if "golden.txt" in l and "Bit Analysis" not in l]
        assert len(data_line) == 1
        assert "Y" in data_line[0].split()[1]  # exact 列
        assert "PASS" in out
        # Level 2: fuzzy/sanity — fp32 级指标
        assert "cosine" in out
        assert "qsnr" in out
        # Level 3: bit analysis — bit 语义分析
        assert "Bit Analysis" in out
        assert "Diff elements:     0" in out
        assert "Bit-exact match" in out
        # 补充: blocked heatmap
        assert "Block Heatmap" in out
        assert "0 failed" in out

    def test_one_block_one_bit_diff(self, tmp_path, capsys):
        """1 block 的 1 个 element 翻 1 bit → 精准定位"""
        data = np.array([0.5, -0.3, 0.1, 0.7] * 16, dtype=np.float32)
        packed_g, _ = quantize(data, "bfpp8")
        packed_r = packed_g.copy()
        # bfpp8: block_size=32, 2 blocks, 前2字节是 shared exp
        # 翻转第3个字节 (第1个 mantissa) 的最低位
        packed_r[2] = packed_r[2] ^ np.int8(1)
        g_path, r_path = _make_packed_pair(tmp_path, packed_g, packed_r)
        ret = cmd_compare(
            action="diff", golden=g_path, result=r_path,
            format="hex_text", qtype="bfpp8", shape="64",
        )
        out = capsys.readouterr().out

        # Level 1: exact = N (字节不同)
        assert ret == 1
        data_line = [l for l in out.split("\n") if "golden.txt" in l and "Bit Analysis" not in l]
        assert "N" in data_line[0].split()[1]
        # Level 2: fuzzy/sanity — fp32 级能看到差异
        assert "cosine" in out
        # Level 3: bit analysis 精准定位到 1 个元素
        assert "Bit Analysis" in out
        assert "Diff elements:     1" in out
        assert "Mantissa-only diff: 1 elements" in out
        assert "indices: [0]" in out
        assert "Sign flips:        0" in out
        # 补充: blocked heatmap 只有 1 个 block 受影响
        assert "Block Heatmap" in out
        assert "1 failed" in out

    def test_half_different(self, tmp_path, capsys):
        """前一半 (+5.0 偏移) → 大面积异常"""
        g_data = np.ones(64, dtype=np.float32)
        r_data = g_data.copy()
        r_data[:32] += 5.0  # 前 32 个元素偏移
        g_path, r_path = _make_hex_pair(tmp_path, g_data, r_data, qtype="bfpp8")
        ret = cmd_compare(
            action="diff", golden=g_path, result=r_path,
            format="hex_text", qtype="bfpp8", shape="64",
        )
        out = capsys.readouterr().out

        # Level 1: exact = N
        assert ret == 1
        data_line = [l for l in out.split("\n") if "golden.txt" in l and "Bit Analysis" not in l]
        assert "N" in data_line[0].split()[1]
        # Level 2: fuzzy/sanity — fp32 级检测到大偏差
        assert "BOTH_SUSPECT" in out or "DUT_ISSUE" in out
        # Level 3: bit analysis — 32 个元素有差异
        assert "Bit Analysis" in out
        assert "Diff elements:     32" in out
        assert "Mantissa-only diff: 32 elements" in out
        # 补充: blocked heatmap — 第一个 block 异常, 第二个正常
        assert "Block Heatmap" in out
        assert "1 failed" in out  # 只有 block 0 失败
