"""
比对模块测试
"""

import numpy as np
import pytest

from aidevtools.compare import (
    CompareConfig,
    CompareEngine,
    CompareResult,
    CompareStatus,
    ExactResult,
    FuzzyResult,
    SanityResult,
    calc_cosine,
    calc_qsnr,
    check_golden_sanity,
    compare_exact,
    compare_full,
    compare_fuzzy,
)


class TestMetrics:
    """指标计算测试"""

    def test_calc_qsnr_identical(self):
        """完全相同数据的 QSNR"""
        data = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        qsnr = calc_qsnr(data, data)
        assert qsnr == float("inf")

    def test_calc_qsnr_small_noise(self):
        """小噪声的 QSNR"""
        golden = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        result = golden + 0.001  # 添加小噪声
        qsnr = calc_qsnr(golden, result)
        assert qsnr > 50  # 应该是高 QSNR

    def test_calc_qsnr_large_noise(self):
        """大噪声的 QSNR"""
        golden = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        result = golden + 1.0  # 添加大噪声
        qsnr = calc_qsnr(golden, result)
        assert qsnr < 20  # QSNR 较低

    def test_calc_cosine_identical(self):
        """完全相同向量的余弦相似度"""
        data = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        cosine = calc_cosine(data, data)
        assert abs(cosine - 1.0) < 1e-6

    def test_calc_cosine_orthogonal(self):
        """正交向量的余弦相似度"""
        a = np.array([1.0, 0.0], dtype=np.float32)
        b = np.array([0.0, 1.0], dtype=np.float32)
        cosine = calc_cosine(a, b)
        assert abs(cosine) < 1e-6

    def test_calc_cosine_opposite(self):
        """相反向量的余弦相似度"""
        a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        b = -a
        cosine = calc_cosine(a, b)
        assert abs(cosine + 1.0) < 1e-6


class TestExactCompare:
    """精确比对测试"""

    def test_exact_match(self):
        """完全匹配"""
        data = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        result = compare_exact(data, data)
        assert result.passed is True
        assert result.mismatch_count == 0
        assert result.max_abs == 0.0

    def test_compare_bit(self):
        """bit 级比对"""
        from aidevtools.compare.exact import compare_bit

        data1 = b"\x00\x01\x02\x03"
        data2 = b"\x00\x01\x02\x03"
        data3 = b"\x00\x01\x02\x04"

        assert compare_bit(data1, data2) is True
        assert compare_bit(data1, data3) is False

    def test_exact_mismatch(self):
        """不匹配"""
        golden = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        result_data = np.array([1.0, 2.5, 3.0], dtype=np.float32)
        result = compare_exact(golden, result_data)
        assert result.passed is False
        assert result.mismatch_count > 0
        assert result.max_abs == 0.5

    def test_exact_with_tolerance(self):
        """允许误差的精确比对"""
        golden = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        result_data = np.array([1.001, 2.001, 3.001], dtype=np.float32)
        result = compare_exact(golden, result_data, max_abs=0.01)
        assert result.passed is True

    def test_exact_with_max_count(self):
        """允许一定数量的不匹配"""
        golden = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        result_data = np.array([1.0, 2.5, 3.0, 4.0], dtype=np.float32)
        result = compare_exact(golden, result_data, max_count=1)
        assert result.passed is True


class TestFuzzyCompare:
    """模糊比对测试"""

    def test_fuzzy_pass(self):
        """通过模糊比对"""
        golden = np.random.randn(100).astype(np.float32)
        result_data = golden + np.random.randn(100).astype(np.float32) * 0.0001
        config = CompareConfig(
            fuzzy_min_qsnr=30.0,
            fuzzy_min_cosine=0.99,
            fuzzy_max_exceed_ratio=0.1,  # 允许 10% 超限
        )
        result = compare_fuzzy(golden, result_data, config)
        assert result.passed is True
        assert result.qsnr > 30.0
        assert result.cosine > 0.99

    def test_fuzzy_fail_low_qsnr(self):
        """QSNR 不足导致失败"""
        golden = np.random.randn(100).astype(np.float32)
        result_data = golden + np.random.randn(100).astype(np.float32) * 0.5
        config = CompareConfig(
            fuzzy_min_qsnr=30.0,
            fuzzy_min_cosine=0.0,  # 不检查余弦
        )
        result = compare_fuzzy(golden, result_data, config)
        assert result.passed is False
        assert result.qsnr < 30.0

    def test_fuzzy_default_config(self):
        """使用默认配置的模糊比对"""
        golden = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        result_data = golden.copy()

        # 不传 config，使用默认配置
        result = compare_fuzzy(golden, result_data)
        assert result.passed is True

    def test_compare_isclose(self):
        """IsClose 比对"""
        from aidevtools.compare.fuzzy import compare_isclose

        golden = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        result_data = np.array([1.001, 2.001, 3.001], dtype=np.float32)

        result = compare_isclose(golden, result_data, atol=0.01, rtol=0.0)
        assert result.passed is True

    def test_compare_isclose_fail(self):
        """IsClose 比对失败"""
        from aidevtools.compare.fuzzy import compare_isclose

        golden = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        result_data = np.array([1.1, 2.1, 3.1], dtype=np.float32)

        result = compare_isclose(golden, result_data, atol=0.01, rtol=0.0)
        assert result.passed is False


class TestSanityCheck:
    """Golden 自检测试"""

    def test_sanity_valid(self):
        """有效的 Golden"""
        golden = np.random.randn(100).astype(np.float32)
        result = check_golden_sanity(golden)
        assert result.valid is True
        assert result.no_nan_inf is True
        assert result.non_zero is True

    def test_sanity_all_zero(self):
        """全零 Golden"""
        golden = np.zeros(100, dtype=np.float32)
        config = CompareConfig(sanity_min_nonzero_ratio=0.01)
        result = check_golden_sanity(golden, config=config)
        assert result.valid is False
        assert result.non_zero is False

    def test_sanity_with_nan(self):
        """包含 NaN 的 Golden"""
        golden = np.array([1.0, np.nan, 3.0], dtype=np.float32)
        result = check_golden_sanity(golden)
        assert result.valid is False
        assert result.no_nan_inf is False

    def test_sanity_with_inf(self):
        """包含 Inf 的 Golden"""
        golden = np.array([1.0, np.inf, 3.0], dtype=np.float32)
        result = check_golden_sanity(golden)
        assert result.valid is False
        assert result.no_nan_inf is False

    def test_sanity_qsnr_check(self):
        """量化 QSNR 检查"""
        golden_pure = np.random.randn(100).astype(np.float32)
        golden_qnt = golden_pure + np.random.randn(100).astype(np.float32) * 0.001
        config = CompareConfig(sanity_min_qsnr=20.0)
        result = check_golden_sanity(golden_pure, golden_qnt, config)
        assert result.valid is True
        assert result.qsnr_valid is True


class TestCompareEngine:
    """比对引擎测试"""

    def test_status_pass(self):
        """PASS 状态: DUT 通过 + Golden 有效"""
        golden = np.random.randn(100).astype(np.float32)
        dut = golden.copy()  # 完全匹配
        golden_qnt = golden + np.random.randn(100).astype(np.float32) * 0.001

        engine = CompareEngine()
        result = engine.compare(dut, golden, golden_qnt)

        assert result.status == CompareStatus.PASS
        assert result.dut_passed is True
        assert result.golden_valid is True

    def test_status_golden_suspect(self):
        """GOLDEN_SUSPECT 状态: DUT 通过 + Golden 可疑"""
        golden_pure = np.random.randn(100).astype(np.float32)
        golden_qnt = golden_pure + np.random.randn(100).astype(np.float32) * 10  # 大差异
        dut = golden_qnt.copy()  # DUT 匹配 golden_qnt

        config = CompareConfig(
            sanity_min_qsnr=30.0,  # 高阈值导致 sanity 失败
        )
        engine = CompareEngine(config)
        result = engine.compare(dut, golden_pure, golden_qnt)

        assert result.status == CompareStatus.GOLDEN_SUSPECT
        assert result.dut_passed is True
        assert result.golden_valid is False

    def test_status_dut_issue(self):
        """DUT_ISSUE 状态: DUT 不通过 + Golden 有效"""
        golden = np.random.randn(100).astype(np.float32)
        dut = golden + np.random.randn(100).astype(np.float32) * 10  # 大误差

        config = CompareConfig(
            fuzzy_min_qsnr=40.0,
            fuzzy_min_cosine=0.999,
        )
        engine = CompareEngine(config)
        result = engine.compare(dut, golden, golden)

        assert result.status == CompareStatus.DUT_ISSUE
        assert result.dut_passed is False
        assert result.golden_valid is True

    def test_status_both_suspect(self):
        """BOTH_SUSPECT 状态: DUT 不通过 + Golden 可疑"""
        golden_pure = np.random.randn(100).astype(np.float32)
        golden_qnt = golden_pure + np.random.randn(100).astype(np.float32) * 10
        dut = np.random.randn(100).astype(np.float32) * 100  # 完全不同

        config = CompareConfig(
            sanity_min_qsnr=40.0,
            fuzzy_min_qsnr=40.0,
        )
        engine = CompareEngine(config)
        result = engine.compare(dut, golden_pure, golden_qnt)

        assert result.status == CompareStatus.BOTH_SUSPECT
        assert result.dut_passed is False
        assert result.golden_valid is False


class TestCompareFull:
    """便捷函数测试"""

    def test_compare_full_basic(self):
        """基本完整比对"""
        golden = np.random.randn(100).astype(np.float32)
        dut = golden.copy()

        result = compare_full(dut, golden)

        assert isinstance(result, CompareResult)
        assert result.status == CompareStatus.PASS


class TestDetermineStatusFunction:
    """独立 determine_status 函数测试"""

    def test_determine_status_pass(self):
        """PASS 状态"""
        from aidevtools.compare.engine import determine_status

        exact = ExactResult(passed=True, mismatch_count=0, first_diff_offset=-1, max_abs=0.0)
        sanity = SanityResult(valid=True)

        status = determine_status(exact, None, None, sanity)
        assert status == CompareStatus.PASS

    def test_determine_status_fuzzy_pass(self):
        """通过 fuzzy_qnt 的 PASS 状态"""
        from aidevtools.compare.engine import determine_status

        exact = ExactResult(passed=False, mismatch_count=1, first_diff_offset=0, max_abs=0.1)
        fuzzy_qnt = FuzzyResult(
            passed=True, max_abs=0.1, mean_abs=0.05,
            max_rel=0.01, qsnr=50.0, cosine=0.999,
            total_elements=100, exceed_count=0
        )
        sanity = SanityResult(valid=True)

        status = determine_status(exact, None, fuzzy_qnt, sanity)
        assert status == CompareStatus.PASS

    def test_determine_status_golden_suspect(self):
        """GOLDEN_SUSPECT 状态"""
        from aidevtools.compare.engine import determine_status

        exact = ExactResult(passed=True, mismatch_count=0, first_diff_offset=-1, max_abs=0.0)
        sanity = SanityResult(valid=False)

        status = determine_status(exact, None, None, sanity)
        assert status == CompareStatus.GOLDEN_SUSPECT

    def test_determine_status_dut_issue(self):
        """DUT_ISSUE 状态"""
        from aidevtools.compare.engine import determine_status

        exact = ExactResult(passed=False, mismatch_count=10, first_diff_offset=0, max_abs=1.0)
        fuzzy_qnt = FuzzyResult(
            passed=False, max_abs=1.0, mean_abs=0.5,
            max_rel=0.1, qsnr=10.0, cosine=0.9,
            total_elements=100, exceed_count=50
        )
        sanity = SanityResult(valid=True)

        status = determine_status(exact, None, fuzzy_qnt, sanity)
        assert status == CompareStatus.DUT_ISSUE

    def test_determine_status_both_suspect(self):
        """BOTH_SUSPECT 状态"""
        from aidevtools.compare.engine import determine_status

        exact = ExactResult(passed=False, mismatch_count=10, first_diff_offset=0, max_abs=1.0)
        sanity = SanityResult(valid=False)

        status = determine_status(exact, None, None, sanity)
        assert status == CompareStatus.BOTH_SUSPECT

    def test_determine_status_no_results(self):
        """无比对结果"""
        from aidevtools.compare.engine import determine_status

        status = determine_status(None, None, None, None)
        assert status == CompareStatus.DUT_ISSUE  # 默认 DUT 不通过, golden 有效


class TestCompareResult:
    """CompareResult 测试"""

    def test_determine_status(self):
        """状态判定"""
        result = CompareResult()

        # DUT pass, Golden valid -> PASS
        result.exact = ExactResult(passed=True, mismatch_count=0, first_diff_offset=-1, max_abs=0.0)
        result.sanity = SanityResult(valid=True)
        assert result.determine_status() == CompareStatus.PASS

        # DUT pass, Golden invalid -> GOLDEN_SUSPECT
        result.sanity = SanityResult(valid=False)
        assert result.determine_status() == CompareStatus.GOLDEN_SUSPECT

        # DUT fail, Golden valid -> DUT_ISSUE
        result.exact = ExactResult(passed=False, mismatch_count=10, first_diff_offset=0, max_abs=1.0)
        result.fuzzy_qnt = FuzzyResult(
            passed=False, max_abs=1.0, mean_abs=0.5, max_rel=0.1,
            qsnr=10.0, cosine=0.9, total_elements=100, exceed_count=50
        )
        result.sanity = SanityResult(valid=True)
        assert result.determine_status() == CompareStatus.DUT_ISSUE

        # DUT fail, Golden invalid -> BOTH_SUSPECT
        result.sanity = SanityResult(valid=False)
        assert result.determine_status() == CompareStatus.BOTH_SUSPECT


class TestMetricsExtended:
    """扩展指标计算测试"""

    def test_calc_abs_error(self):
        """绝对误差计算"""
        from aidevtools.compare.metrics import calc_abs_error

        golden = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        result = np.array([1.1, 2.2, 3.3], dtype=np.float32)

        max_abs, mean_abs, abs_err = calc_abs_error(golden, result)
        assert abs(max_abs - 0.3) < 0.01
        assert abs(mean_abs - 0.2) < 0.01
        assert len(abs_err) == 3

    def test_calc_abs_error_empty(self):
        """空数组绝对误差"""
        from aidevtools.compare.metrics import calc_abs_error

        golden = np.array([], dtype=np.float32)
        result = np.array([], dtype=np.float32)

        max_abs, mean_abs, _ = calc_abs_error(golden, result)
        assert max_abs == 0.0
        assert mean_abs == 0.0

    def test_calc_rel_error(self):
        """相对误差计算"""
        from aidevtools.compare.metrics import calc_rel_error

        golden = np.array([1.0, 2.0, 4.0], dtype=np.float32)
        result = np.array([1.1, 2.2, 4.4], dtype=np.float32)

        max_rel, mean_rel, rel_err = calc_rel_error(golden, result)
        assert abs(max_rel - 0.1) < 0.01  # 10% relative error
        assert len(rel_err) == 3

    def test_calc_rel_error_zero_golden(self):
        """Golden 包含零时的相对误差"""
        from aidevtools.compare.metrics import calc_rel_error

        golden = np.array([0.0, 1.0, 2.0], dtype=np.float32)
        result = np.array([0.1, 1.1, 2.2], dtype=np.float32)

        max_rel, mean_rel, rel_err = calc_rel_error(golden, result)
        # 零处相对误差为 0
        assert rel_err[0] == 0.0

    def test_calc_rel_error_empty(self):
        """空数组相对误差"""
        from aidevtools.compare.metrics import calc_rel_error

        golden = np.array([], dtype=np.float32)
        result = np.array([], dtype=np.float32)

        max_rel, mean_rel, _ = calc_rel_error(golden, result)
        assert max_rel == 0.0
        assert mean_rel == 0.0

    def test_calc_exceed_count(self):
        """超阈值计数"""
        from aidevtools.compare.metrics import calc_exceed_count

        golden = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        result = np.array([1.0, 2.5, 3.0, 5.0], dtype=np.float32)

        count = calc_exceed_count(golden, result, atol=0.1, rtol=0.0)
        assert count == 2  # 2.0 vs 2.5 和 4.0 vs 5.0

    def test_calc_exceed_count_with_rtol(self):
        """带相对容差的超阈值计数"""
        from aidevtools.compare.metrics import calc_exceed_count

        golden = np.array([1.0, 10.0, 100.0], dtype=np.float32)
        result = np.array([1.05, 10.5, 105.0], dtype=np.float32)

        # atol=0.0, rtol=0.1 允许 10% 相对误差
        # 误差分别为 5%, 5%, 5%，都在 10% 以内
        count = calc_exceed_count(golden, result, atol=0.0, rtol=0.1)
        assert count == 0

    def test_check_nan_inf(self):
        """NaN/Inf 检查"""
        from aidevtools.compare.metrics import check_nan_inf

        data = np.array([1.0, np.nan, np.inf, -np.inf, 2.0], dtype=np.float32)
        nan_count, inf_count, total = check_nan_inf(data)

        assert nan_count == 1
        assert inf_count == 2
        assert total == 5

    def test_check_nan_inf_clean(self):
        """无 NaN/Inf 数据"""
        from aidevtools.compare.metrics import check_nan_inf

        data = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        nan_count, inf_count, total = check_nan_inf(data)

        assert nan_count == 0
        assert inf_count == 0
        assert total == 3

    def test_check_nonzero(self):
        """非零检查"""
        from aidevtools.compare.metrics import check_nonzero

        data = np.array([0.0, 1.0, 0.0, 2.0, 0.0], dtype=np.float32)
        nonzero_count, total, ratio = check_nonzero(data)

        assert nonzero_count == 2
        assert total == 5
        assert abs(ratio - 0.4) < 0.01

    def test_check_nonzero_all_zero(self):
        """全零数据"""
        from aidevtools.compare.metrics import check_nonzero

        data = np.zeros(10, dtype=np.float32)
        nonzero_count, total, ratio = check_nonzero(data)

        assert nonzero_count == 0
        assert ratio == 0.0

    def test_calc_cosine_zero_vector(self):
        """零向量的余弦相似度"""
        a = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        b = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        cosine = calc_cosine(a, b)
        assert cosine == 0.0


class TestSanityExtended:
    """扩展 Golden 自检测试"""

    def test_check_data_sanity_valid(self):
        """有效数据自检"""
        from aidevtools.compare.sanity import check_data_sanity

        data = np.random.randn(100).astype(np.float32)
        result = check_data_sanity(data, name="test_data")

        assert result.valid is True
        assert result.non_zero is True
        assert result.no_nan_inf is True

    def test_check_data_sanity_all_zero(self):
        """全零数据自检"""
        from aidevtools.compare.sanity import check_data_sanity

        data = np.zeros(100, dtype=np.float32)
        result = check_data_sanity(data, name="zero_data")

        assert result.valid is False
        assert result.non_zero is False
        assert "零" in result.messages[0]

    def test_check_data_sanity_with_nan(self):
        """含 NaN 数据自检"""
        from aidevtools.compare.sanity import check_data_sanity

        data = np.array([1.0, np.nan, 3.0], dtype=np.float32)
        result = check_data_sanity(data, name="nan_data")

        assert result.valid is False
        assert result.no_nan_inf is False
        assert "NaN" in result.messages[0]

    def test_check_data_sanity_with_inf(self):
        """含 Inf 数据自检"""
        from aidevtools.compare.sanity import check_data_sanity

        data = np.array([1.0, np.inf, 3.0], dtype=np.float32)
        result = check_data_sanity(data, name="inf_data")

        assert result.valid is False
        assert "Inf" in result.messages[0]

    def test_sanity_constant_golden(self):
        """常数 Golden 检查"""
        golden = np.ones(100, dtype=np.float32) * 5.0
        result = check_golden_sanity(golden)

        assert result.valid is False
        assert result.range_valid is False

    def test_sanity_low_qsnr(self):
        """低 QSNR 的量化 Golden"""
        golden_pure = np.random.randn(100).astype(np.float32)
        golden_qnt = golden_pure + np.random.randn(100).astype(np.float32) * 5
        config = CompareConfig(sanity_min_qsnr=40.0)

        result = check_golden_sanity(golden_pure, golden_qnt, config)

        assert result.valid is False
        assert result.qsnr_valid is False


class TestEngineExtended:
    """扩展比对引擎测试"""

    def test_compare_exact_only(self):
        """仅精确比对"""
        golden = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        dut = golden.copy()

        engine = CompareEngine()
        result = engine.compare_exact_only(dut, golden, name="test_exact")

        assert result.status == CompareStatus.PASS
        assert result.exact is not None
        assert result.exact.passed is True
        assert result.name == "test_exact"

    def test_compare_exact_only_fail(self):
        """仅精确比对 - 失败"""
        golden = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        dut = np.array([1.0, 2.5, 3.0], dtype=np.float32)

        engine = CompareEngine()
        result = engine.compare_exact_only(dut, golden)

        assert result.status == CompareStatus.DUT_ISSUE
        assert result.exact.passed is False

    def test_compare_fuzzy_only(self):
        """仅模糊比对"""
        golden = np.random.randn(100).astype(np.float32)
        dut = golden + np.random.randn(100).astype(np.float32) * 0.0001

        config = CompareConfig(
            fuzzy_min_qsnr=30.0,
            fuzzy_min_cosine=0.99,
            fuzzy_max_exceed_ratio=0.1,  # 允许 10% 超限
        )
        engine = CompareEngine(config)
        result = engine.compare_fuzzy_only(dut, golden, name="test_fuzzy")

        assert result.status == CompareStatus.PASS
        assert result.fuzzy_qnt is not None
        assert result.fuzzy_qnt.passed is True
        assert result.name == "test_fuzzy"

    def test_compare_fuzzy_only_fail(self):
        """仅模糊比对 - 失败"""
        golden = np.random.randn(100).astype(np.float32)
        dut = np.random.randn(100).astype(np.float32) * 10

        config = CompareConfig(fuzzy_min_qsnr=40.0, fuzzy_min_cosine=0.999)
        engine = CompareEngine(config)
        result = engine.compare_fuzzy_only(dut, golden)

        assert result.status == CompareStatus.DUT_ISSUE
        assert result.fuzzy_qnt.passed is False

    def test_compare_with_name_and_op_id(self):
        """带名称和 op_id 的比对"""
        golden = np.random.randn(50).astype(np.float32)
        dut = golden.copy()

        engine = CompareEngine()
        result = engine.compare(dut, golden, name="conv1", op_id=42)

        assert result.name == "conv1"
        assert result.op_id == 42


class TestReportGeneration:
    """报告生成测试"""

    def test_format_result_row_pass(self):
        """格式化通过结果行"""
        from aidevtools.compare.report import format_result_row

        result = CompareResult(name="test_op", op_id=1, status=CompareStatus.PASS)
        result.exact = ExactResult(passed=True, mismatch_count=0, first_diff_offset=-1, max_abs=0.0)
        result.fuzzy_pure = FuzzyResult(
            passed=True, max_abs=0.001, mean_abs=0.0005,
            max_rel=0.01, qsnr=50.0, cosine=0.9999,
            total_elements=100, exceed_count=0
        )
        result.fuzzy_qnt = FuzzyResult(
            passed=True, max_abs=0.001, mean_abs=0.0005,
            max_rel=0.01, qsnr=50.0, cosine=0.9999,
            total_elements=100, exceed_count=0
        )
        result.sanity = SanityResult(valid=True)

        row = format_result_row(result)
        assert "test_op" in row
        assert "PASS" in row
        assert "Y" in row

    def test_format_result_row_fail(self):
        """格式化失败结果行"""
        from aidevtools.compare.report import format_result_row

        result = CompareResult(name="", op_id=5, status=CompareStatus.DUT_ISSUE)
        result.exact = ExactResult(passed=False, mismatch_count=10, first_diff_offset=0, max_abs=1.0)

        row = format_result_row(result)
        assert "op_5" in row  # 使用 op_id
        assert "DUT_ISSUE" in row
        assert "N" in row

    def test_format_result_row_inf_qsnr(self):
        """格式化 inf QSNR 结果行"""
        from aidevtools.compare.report import format_result_row

        result = CompareResult(name="test", status=CompareStatus.PASS)
        result.fuzzy_qnt = FuzzyResult(
            passed=True, max_abs=0.0, mean_abs=0.0,
            max_rel=0.0, qsnr=float("inf"), cosine=1.0,
            total_elements=100, exceed_count=0
        )

        row = format_result_row(result)
        assert "inf" in row

    def test_print_compare_table(self, capsys):
        """打印比对表格"""
        from aidevtools.compare.report import print_compare_table

        results = [
            CompareResult(name="op1", status=CompareStatus.PASS),
            CompareResult(name="op2", status=CompareStatus.DUT_ISSUE),
        ]
        results[0].sanity = SanityResult(valid=True)
        results[1].sanity = SanityResult(valid=True)

        print_compare_table(results)
        captured = capsys.readouterr()

        assert "op1" in captured.out
        assert "op2" in captured.out
        assert "PASS" in captured.out
        assert "Summary" in captured.out
        assert "total: 2" in captured.out

    def test_generate_text_report(self):
        """生成文本报告"""
        from aidevtools.compare.report import generate_text_report

        results = [
            CompareResult(name="test_op", status=CompareStatus.PASS),
        ]
        results[0].exact = ExactResult(passed=True, mismatch_count=0, first_diff_offset=-1, max_abs=0.0)
        results[0].fuzzy_qnt = FuzzyResult(
            passed=True, max_abs=0.001, mean_abs=0.0005,
            max_rel=0.01, qsnr=50.0, cosine=0.9999,
            total_elements=100, exceed_count=0
        )
        results[0].sanity = SanityResult(valid=True, checks={"non_zero": True}, messages=["OK"])

        report = generate_text_report(results)

        assert "Compare Report" in report
        assert "test_op" in report
        assert "PASS" in report
        assert "Exact Compare" in report
        assert "Fuzzy Compare" in report
        assert "Golden Sanity" in report
        assert "Summary" in report

    def test_generate_text_report_to_file(self, tmp_path):
        """生成文本报告到文件"""
        from aidevtools.compare.report import generate_text_report

        results = [
            CompareResult(name="test", status=CompareStatus.PASS),
        ]

        output_path = tmp_path / "report.txt"
        generate_text_report(results, str(output_path))

        assert output_path.exists()
        content = output_path.read_text()
        assert "Compare Report" in content

    def test_generate_json_report(self):
        """生成 JSON 报告"""
        from aidevtools.compare.report import generate_json_report

        results = [
            CompareResult(name="test_op", op_id=1, status=CompareStatus.PASS),
        ]
        results[0].exact = ExactResult(passed=True, mismatch_count=0, first_diff_offset=-1, max_abs=0.0)
        results[0].fuzzy_pure = FuzzyResult(
            passed=True, max_abs=0.001, mean_abs=0.0005,
            max_rel=0.01, qsnr=50.0, cosine=0.9999,
            total_elements=100, exceed_count=0
        )
        results[0].fuzzy_qnt = FuzzyResult(
            passed=True, max_abs=0.001, mean_abs=0.0005,
            max_rel=0.01, qsnr=50.0, cosine=0.9999,
            total_elements=100, exceed_count=0
        )
        results[0].sanity = SanityResult(valid=True, checks={"test": True}, messages=["msg"])

        report = generate_json_report(results)

        assert "results" in report
        assert "summary" in report
        assert len(report["results"]) == 1
        assert report["results"][0]["name"] == "test_op"
        assert report["results"][0]["exact"]["passed"] is True
        assert report["results"][0]["fuzzy_pure"]["qsnr"] == 50.0
        assert report["results"][0]["fuzzy_qnt"]["cosine"] == 0.9999
        assert report["results"][0]["sanity"]["valid"] is True
        assert report["summary"]["total"] == 1
        assert report["summary"]["by_status"]["PASS"] == 1

    def test_generate_json_report_to_file(self, tmp_path):
        """生成 JSON 报告到文件"""
        import json
        from aidevtools.compare.report import generate_json_report

        results = [
            CompareResult(name="test", status=CompareStatus.DUT_ISSUE),
        ]

        output_path = tmp_path / "report.json"
        generate_json_report(results, str(output_path))

        assert output_path.exists()
        content = json.loads(output_path.read_text())
        assert content["summary"]["by_status"]["DUT_ISSUE"] == 1

    def test_generate_json_report_multiple_status(self):
        """生成多状态 JSON 报告"""
        from aidevtools.compare.report import generate_json_report

        results = [
            CompareResult(name="op1", status=CompareStatus.PASS),
            CompareResult(name="op2", status=CompareStatus.DUT_ISSUE),
            CompareResult(name="op3", status=CompareStatus.GOLDEN_SUSPECT),
            CompareResult(name="op4", status=CompareStatus.BOTH_SUSPECT),
        ]

        report = generate_json_report(results)

        assert report["summary"]["total"] == 4
        assert report["summary"]["by_status"]["PASS"] == 1
        assert report["summary"]["by_status"]["DUT_ISSUE"] == 1
        assert report["summary"]["by_status"]["GOLDEN_SUSPECT"] == 1
        assert report["summary"]["by_status"]["BOTH_SUSPECT"] == 1
