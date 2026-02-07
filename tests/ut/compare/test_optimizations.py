"""比对优化功能测试

测试:
1. calc_all_metrics - 单次遍历计算
2. calc_all_metrics_early_exit - 带 early exit 的计算
3. compare_blocked + print_block_heatmap + find_worst_blocks - 分块比对
4. tools/compare/diff.py 统一委托后的兼容性
"""

import numpy as np
import pytest

from aidevtools.compare.metrics import (
    AllMetrics,
    calc_all_metrics,
    calc_all_metrics_early_exit,
    calc_qsnr,
    calc_cosine,
)
from aidevtools.compare.blocked import (
    BlockResult,
    compare_blocked,
    print_block_heatmap,
    find_worst_blocks,
)


class TestCalcAllMetrics:
    """单次遍历计算测试"""

    def test_identical_arrays(self):
        """相同数据: QSNR=inf, cosine=1.0, 误差=0"""
        data = np.random.RandomState(42).randn(256).astype(np.float32)
        m = calc_all_metrics(data, data)

        assert isinstance(m, AllMetrics)
        assert m.qsnr == float("inf")
        assert m.cosine == pytest.approx(1.0, abs=1e-10)
        assert m.max_abs == 0.0
        assert m.mean_abs == 0.0
        assert m.exceed_count == 0
        assert m.total_elements == 256

    def test_small_noise(self):
        """小噪声: 高 QSNR, 高 cosine"""
        rng = np.random.RandomState(42)
        golden = rng.randn(1000).astype(np.float32)
        result = golden + rng.randn(1000).astype(np.float32) * 1e-5

        m = calc_all_metrics(golden, result)

        assert m.qsnr > 80
        assert m.cosine > 0.9999
        assert m.max_abs < 1e-4
        assert m.total_elements == 1000

    def test_large_noise(self):
        """大噪声: 低 QSNR"""
        rng = np.random.RandomState(42)
        golden = rng.randn(1000).astype(np.float32)
        result = golden + rng.randn(1000).astype(np.float32) * 0.5

        m = calc_all_metrics(golden, result)

        assert m.qsnr < 20
        assert m.cosine < 0.999
        assert m.max_abs > 0.1

    def test_exceed_count(self):
        """超阈值元素数计算"""
        golden = np.ones(100, dtype=np.float32)
        result = golden.copy()
        result[:10] += 0.1  # 10 个元素超限

        m = calc_all_metrics(golden, result, atol=0.01, rtol=0.0)

        assert m.exceed_count == 10

    def test_empty_array(self):
        """空数组"""
        m = calc_all_metrics(np.array([]), np.array([]))

        assert m.qsnr == float("inf")
        assert m.total_elements == 0
        assert m.exceed_count == 0

    def test_consistency_with_standalone(self):
        """与独立函数结果一致"""
        rng = np.random.RandomState(123)
        golden = rng.randn(500).astype(np.float32)
        result = golden + rng.randn(500).astype(np.float32) * 0.01

        m = calc_all_metrics(golden, result)
        qsnr_standalone = calc_qsnr(golden, result)
        cosine_standalone = calc_cosine(golden, result)

        assert m.qsnr == pytest.approx(qsnr_standalone, rel=1e-10)
        assert m.cosine == pytest.approx(cosine_standalone, rel=1e-10)

    def test_multidim_array(self):
        """多维数组自动 flatten"""
        rng = np.random.RandomState(42)
        golden = rng.randn(2, 3, 4).astype(np.float32)
        result = golden + rng.randn(2, 3, 4).astype(np.float32) * 1e-5

        m = calc_all_metrics(golden, result)

        assert m.total_elements == 24


class TestCalcAllMetricsEarlyExit:
    """带 early exit 的计算测试"""

    def test_identical_arrays(self):
        """相同数据应全部通过"""
        data = np.random.RandomState(42).randn(256).astype(np.float32)
        m = calc_all_metrics_early_exit(
            data, data,
            min_qsnr=30.0, min_cosine=0.999, max_exceed_ratio=0.0,
        )

        assert m.qsnr == float("inf")
        assert m.cosine == pytest.approx(1.0, abs=1e-10)
        assert m.exceed_count == 0

    def test_consistency_with_non_early_exit(self):
        """与非 early-exit 版本结果一致"""
        rng = np.random.RandomState(42)
        golden = rng.randn(500).astype(np.float32)
        result = golden + rng.randn(500).astype(np.float32) * 0.01

        m1 = calc_all_metrics(golden, result, atol=1e-5, rtol=1e-3)
        m2 = calc_all_metrics_early_exit(
            golden, result,
            atol=1e-5, rtol=1e-3,
            min_qsnr=0.0, min_cosine=0.0, max_exceed_ratio=1.0,
        )

        assert m1.qsnr == pytest.approx(m2.qsnr, rel=1e-10)
        assert m1.cosine == pytest.approx(m2.cosine, rel=1e-10)
        assert m1.max_abs == pytest.approx(m2.max_abs, rel=1e-10)
        assert m1.exceed_count == m2.exceed_count

    def test_early_exit_cosine_fail(self):
        """cosine 不达标时仍返回完整结果"""
        rng = np.random.RandomState(42)
        golden = rng.randn(100).astype(np.float32)
        # 完全不同的数据 → cosine 很低
        result = rng.randn(100).astype(np.float32) * 10

        m = calc_all_metrics_early_exit(
            golden, result,
            min_qsnr=30.0, min_cosine=0.999, max_exceed_ratio=0.0,
        )

        assert m.cosine < 0.999  # 确认 early exit 触发条件
        assert m.total_elements == 100
        # 即使 early exit, 所有字段仍有值
        assert isinstance(m.qsnr, float)
        assert isinstance(m.max_rel, float)

    def test_early_exit_exceed_fail(self):
        """exceed 不达标时仍返回完整结果"""
        golden = np.ones(100, dtype=np.float32)
        result = golden.copy()
        result[:50] += 1.0  # 50% 元素超限

        m = calc_all_metrics_early_exit(
            golden, result,
            atol=0.01, rtol=0.0,
            min_qsnr=0.0, min_cosine=0.0, max_exceed_ratio=0.1,
        )

        assert m.exceed_count == 50
        assert isinstance(m.qsnr, float)

    def test_empty_array(self):
        """空数组"""
        m = calc_all_metrics_early_exit(
            np.array([]), np.array([]),
            min_qsnr=30.0, min_cosine=0.999, max_exceed_ratio=0.0,
        )

        assert m.total_elements == 0


class TestCompareBlocked:
    """分块比对测试"""

    def test_basic_blocking(self):
        """基本分块"""
        rng = np.random.RandomState(42)
        golden = rng.randn(4096).astype(np.float32)
        result = golden + rng.randn(4096).astype(np.float32) * 1e-6

        blocks = compare_blocked(golden, result, block_size=1024)

        assert len(blocks) == 4
        assert all(isinstance(b, BlockResult) for b in blocks)
        assert all(b.passed for b in blocks)
        assert blocks[0].offset == 0
        assert blocks[0].size == 1024
        assert blocks[1].offset == 1024

    def test_partial_last_block(self):
        """最后一个 block 不满"""
        golden = np.ones(3000, dtype=np.float32)
        result = golden.copy()

        blocks = compare_blocked(golden, result, block_size=1024)

        assert len(blocks) == 3
        assert blocks[0].size == 1024
        assert blocks[1].size == 1024
        assert blocks[2].size == 952  # 3000 - 2048

    def test_failing_blocks(self):
        """有失败的 block"""
        golden = np.ones(2048, dtype=np.float32)
        result = golden.copy()
        # 让第二个 block 有大误差
        result[1024:2048] += 10.0

        blocks = compare_blocked(golden, result, block_size=1024, min_qsnr=30.0)

        assert len(blocks) == 2
        assert blocks[0].passed is True
        assert blocks[1].passed is False
        assert blocks[1].qsnr < 30.0

    def test_custom_thresholds(self):
        """自定义阈值"""
        rng = np.random.RandomState(42)
        golden = rng.randn(2048).astype(np.float32)
        result = golden + rng.randn(2048).astype(np.float32) * 0.01

        # 非常严格的阈值
        blocks_strict = compare_blocked(golden, result, block_size=1024, min_qsnr=60.0)
        # 宽松阈值
        blocks_relaxed = compare_blocked(golden, result, block_size=1024, min_qsnr=10.0)

        # 严格阈值下可能有失败
        strict_pass = sum(1 for b in blocks_strict if b.passed)
        relaxed_pass = sum(1 for b in blocks_relaxed if b.passed)

        assert relaxed_pass >= strict_pass


class TestPrintBlockHeatmap:
    """热力图输出测试"""

    def test_heatmap_output(self, capsys):
        """验证热力图输出格式"""
        blocks = [
            BlockResult(offset=i * 1024, size=1024, qsnr=50.0, cosine=0.999,
                       max_abs=1e-6, exceed_count=0, passed=True)
            for i in range(20)
        ]
        blocks[5] = BlockResult(
            offset=5 * 1024, size=1024, qsnr=15.0, cosine=0.9,
            max_abs=0.1, exceed_count=10, passed=False,
        )

        print_block_heatmap(blocks, cols=10, show_legend=True)

        captured = capsys.readouterr()
        assert "Block Heatmap" in captured.out
        assert "20 blocks" in captured.out
        assert "1 failed" in captured.out
        assert "Legend:" in captured.out

    def test_heatmap_chars(self, capsys):
        """验证热力图字符"""
        blocks = [
            BlockResult(offset=0, size=1024, qsnr=50.0, cosine=1.0,
                       max_abs=0, exceed_count=0, passed=True),  # .
            BlockResult(offset=1024, size=1024, qsnr=25.0, cosine=0.99,
                       max_abs=0.01, exceed_count=0, passed=True),  # o
            BlockResult(offset=2048, size=1024, qsnr=12.0, cosine=0.9,
                       max_abs=0.1, exceed_count=5, passed=False),  # X
            BlockResult(offset=3072, size=1024, qsnr=5.0, cosine=0.5,
                       max_abs=1.0, exceed_count=100, passed=False),  # #
        ]

        print_block_heatmap(blocks, cols=40, show_legend=False)

        captured = capsys.readouterr()
        assert ".oX#" in captured.out


class TestFindWorstBlocks:
    """找最差 block 测试"""

    def test_find_worst(self):
        """找到最差的 N 个 block"""
        blocks = [
            BlockResult(offset=i * 1024, size=1024, qsnr=float(50 - i * 5),
                       cosine=0.99, max_abs=0.01, exceed_count=0,
                       passed=float(50 - i * 5) >= 30.0)
            for i in range(10)
        ]

        worst = find_worst_blocks(blocks, top_n=3)

        assert len(worst) == 3
        # 按 QSNR 从低到高排序
        assert worst[0].qsnr <= worst[1].qsnr <= worst[2].qsnr
        # 最差的是 index 9 (qsnr=5.0)
        assert worst[0].qsnr == 5.0

    def test_find_worst_fewer_blocks(self):
        """block 数少于 top_n"""
        blocks = [
            BlockResult(offset=0, size=1024, qsnr=30.0, cosine=0.99,
                       max_abs=0.01, exceed_count=0, passed=True),
        ]

        worst = find_worst_blocks(blocks, top_n=5)

        assert len(worst) == 1


class TestToolsDiffUnification:
    """验证 tools/compare/diff.py 统一后的兼容性"""

    def test_calc_qsnr_delegated(self):
        """calc_qsnr 委托给核心模块"""
        from aidevtools.tools.compare.diff import calc_qsnr as tools_calc_qsnr
        from aidevtools.compare.metrics import calc_qsnr as core_calc_qsnr

        data = np.random.RandomState(42).randn(100).astype(np.float32)
        noisy = data + np.random.RandomState(0).randn(100).astype(np.float32) * 0.01

        # 应该是同一个函数
        assert tools_calc_qsnr(data, noisy) == core_calc_qsnr(data, noisy)

    def test_calc_cosine_delegated(self):
        """calc_cosine 委托给核心模块"""
        from aidevtools.tools.compare.diff import calc_cosine as tools_calc_cosine
        from aidevtools.compare.metrics import calc_cosine as core_calc_cosine

        data = np.random.RandomState(42).randn(100).astype(np.float32)

        assert tools_calc_cosine(data, data) == core_calc_cosine(data, data)

    def test_compare_bit_delegated(self):
        """compare_bit 委托给核心模块"""
        from aidevtools.tools.compare.diff import compare_bit as tools_compare_bit

        assert tools_compare_bit(b"\x01\x02", b"\x01\x02") is True
        assert tools_compare_bit(b"\x01\x02", b"\x01\x03") is False

    def test_compare_full_uses_single_pass(self):
        """compare_full 使用单次遍历计算"""
        from aidevtools.tools.compare.diff import compare_full, DiffResult

        rng = np.random.RandomState(42)
        golden = rng.randn(1000).astype(np.float32)
        result = golden + rng.randn(1000).astype(np.float32) * 1e-6

        r = compare_full(golden, result)

        assert isinstance(r, DiffResult)
        assert r.passed is True
        assert r.qsnr > 80
        assert r.cosine > 0.999
        assert r.total_elements == 1000

    def test_compare_exact_delegated(self):
        """compare_exact 委托给核心模块"""
        from aidevtools.tools.compare.diff import compare_exact, ExactResult

        golden = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        result = golden.copy()

        r = compare_exact(golden, result)

        assert isinstance(r, ExactResult)
        assert r.passed is True
        assert r.mismatch_count == 0

    def test_compare_block_uses_calc_all_metrics(self):
        """compare_block 使用核心模块的 calc_all_metrics"""
        from aidevtools.tools.compare.diff import compare_block

        golden = np.random.RandomState(42).randn(256).astype(np.float32)
        result = golden + np.random.RandomState(0).randn(256).astype(np.float32) * 1e-6

        blocks = compare_block(golden, result, block_size=256)

        assert len(blocks) > 0
        assert all(isinstance(b, dict) for b in blocks)
        assert all("qsnr" in b for b in blocks)

    def test_compare_isclose_backward_compat(self):
        """compare_isclose 保持向后兼容"""
        from aidevtools.tools.compare.diff import compare_isclose, IsCloseResult

        golden = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        result = golden.copy()

        r = compare_isclose(golden, result, atol=1e-5, rtol=1e-3)

        assert isinstance(r, IsCloseResult)
        assert r.passed is True
        assert r.exceed_count == 0
        assert r.atol == 1e-5
        assert r.rtol == 1e-3

    def test_compare_isclose_shape_mismatch(self):
        """compare_isclose shape 不匹配仍抛 ValueError"""
        from aidevtools.tools.compare.diff import compare_isclose

        golden = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        result = np.array([1.0, 2.0], dtype=np.float32)

        with pytest.raises(ValueError, match="Shape mismatch"):
            compare_isclose(golden, result)
