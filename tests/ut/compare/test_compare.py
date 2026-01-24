"""Compare 模块测试"""
import numpy as np

from aidevtools.tools.compare.diff import compare_bit, compare_block, compare_full, calc_qsnr
from aidevtools.tools.compare.report import gen_heatmap_svg

class TestDiff:
    """比对逻辑测试"""

    def test_compare_bit_pass(self):
        """bit 级 - 通过"""
        data = b'\x01\x02\x03\x04'
        assert compare_bit(data, data) is True

    def test_compare_bit_fail(self):
        """bit 级 - 失败"""
        a = b'\x01\x02\x03\x04'
        b = b'\x01\x02\x03\x05'
        assert compare_bit(a, b) is False

    def test_compare_full_pass(self, golden_result, sim_result):
        """完整级 - 通过"""
        result = compare_full(golden_result, sim_result, atol=1e-5)
        assert result.passed is True
        assert result.qsnr > 40

    def test_compare_full_fail(self, golden_result):
        """完整级 - 失败"""
        bad_result = golden_result + 0.1
        result = compare_full(golden_result, bad_result, atol=1e-5)
        assert result.passed is False

    def test_compare_block(self, golden_result, sim_result):
        """分块级"""
        blocks = compare_block(golden_result, sim_result, block_size=256)
        assert len(blocks) > 0
        assert all(b["passed"] for b in blocks)

    def test_calc_qsnr_identical(self):
        """QSNR - 相同数据"""
        data = np.random.randn(100).astype(np.float32)
        qsnr = calc_qsnr(data, data)
        assert qsnr == float('inf')

    def test_calc_qsnr_noisy(self):
        """QSNR - 带噪声"""
        golden = np.random.randn(100).astype(np.float32)
        noisy = golden + np.random.randn(100).astype(np.float32) * 0.01
        qsnr = calc_qsnr(golden, noisy)
        assert 20 < qsnr < 60

class TestReport:
    """报告生成测试"""

    def test_gen_heatmap_svg(self, tmp_workspace):
        """SVG 热力图"""
        blocks = [
            {"offset": i * 256, "size": 256, "max_abs": 1e-6, "qsnr": 50, "passed": True}
            for i in range(100)
        ]
        blocks[50]["passed"] = False
        blocks[50]["qsnr"] = 15

        svg_path = tmp_workspace / "heatmap.svg"
        gen_heatmap_svg(blocks, str(svg_path))

        assert svg_path.exists()
        content = svg_path.read_text()
        assert "<svg" in content
        assert "#4caf50" in content  # 绿色
        assert "#f44336" in content  # 红色
