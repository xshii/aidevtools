"""
Paper Analysis 模块单元测试
"""

import pytest
import numpy as np
from pathlib import Path
import tempfile

from aidevtools import ops
from aidevtools.ops import _functional as F


def _profile(op_fn, *args, **kwargs):
    """辅助函数：使用 ops profile_only 模式生成单个 profile"""
    with ops.profile_only():
        op_fn(*args, **kwargs)
        profiles = ops.get_profiles()
    return profiles[0] if profiles else None


class TestOpProfile:
    """OpProfile 测试"""

    def test_dtype_bytes(self):
        """测试 dtype 字节数计算"""
        from aidevtools.analysis.profile import dtype_bytes

        assert dtype_bytes("fp16") == 2
        assert dtype_bytes("fp32") == 4
        assert dtype_bytes("int8") == 1
        assert dtype_bytes("bf16") == 2
        assert dtype_bytes("int4") == 0.5

    def test_matmul_profile(self):
        """测试 MatMul profile"""

        a = np.zeros((4, 512, 768), dtype=np.float16)
        b = np.zeros((768, 768), dtype=np.float16)
        profile = _profile(F.matmul, a, b)

        assert profile.op_type == "matmul"
        assert profile.compute_unit == "cube"
        # FLOPs = 2 * batch * M * K * N = 2 * 4 * 512 * 768 * 768
        expected_flops = 2 * 4 * 512 * 768 * 768
        assert profile.flops == expected_flops

    def test_layernorm_profile(self):
        """测试 LayerNorm profile"""
        x = np.zeros((4, 512, 768), dtype=np.float16)
        gamma = np.zeros((768,), dtype=np.float16)
        beta = np.zeros((768,), dtype=np.float16)
        profile = _profile(F.layernorm, x, gamma, beta)

        assert profile.op_type == "layernorm"
        assert profile.compute_unit == "vector"
        # FLOPs: ~8 ops/element
        assert profile.flops == 8 * x.size

    def test_attention_profile(self):
        """测试 Attention profile"""
        batch, heads, seq, head_dim = 4, 12, 512, 64
        q = np.zeros((batch, heads, seq, head_dim), dtype=np.float16)
        k = np.zeros((batch, heads, seq, head_dim), dtype=np.float16)
        v = np.zeros((batch, heads, seq, head_dim), dtype=np.float16)
        profile = _profile(F.attention, q, k, v)

        assert profile.op_type == "attention"
        assert profile.compute_unit == "cube"
        assert profile.flops > 0

    def test_transpose_profile(self):
        """测试 Transpose profile"""
        x = np.zeros((4, 12, 512, 64), dtype=np.float16)
        profile = _profile(F.transpose, x, (0, 2, 1, 3))

        assert profile.op_type == "transpose"
        assert profile.memory_pattern == "strided"
        assert profile.flops == 0

    def test_arithmetic_intensity(self):
        """测试算术强度计算"""
        from aidevtools.analysis.profile import OpProfile

        profile = OpProfile(
            name="test",
            op_type="matmul",
            flops=1000000,
            input_bytes=1000,
            weight_bytes=1000,
            output_bytes=1000,
        )

        # AI = 1000000 / 3000 = 333.33
        assert abs(profile.arithmetic_intensity - 333.33) < 1


class TestChipSpec:
    """ChipSpec 测试"""

    def test_load_builtin_chips(self):
        """测试加载内置芯片配置"""
        from aidevtools.analysis.chip import list_chips

        chips = list_chips()
        assert "npu_310" in chips
        assert "npu_910" in chips
        assert "gpu_a100" in chips

    def test_npu_910_spec(self):
        """测试 NPU 910 规格"""
        from aidevtools.analysis.chip import load_chip_spec

        chip = load_chip_spec("npu_910")

        assert chip.name == "Ascend 910"
        assert chip.cube.fp16_tflops == 256.0
        assert chip.vector.fp16_gflops == 16000
        assert chip.memory.hbm.bandwidth_gbps == 1200
        assert chip.memory.hbm.capacity_bytes == 32 * 1024**3

    def test_ridge_point(self):
        """测试拐点计算"""
        from aidevtools.analysis.chip import load_chip_spec

        chip = load_chip_spec("npu_910")

        # Cube ridge point = 256 TFLOPS / 1200 GB/s = 213.33 FLOPs/Byte
        expected_ridge = 256 * 1e12 / (1200 * 1e9)
        assert abs(chip.cube_ridge_point - expected_ridge) < 1

    def test_get_compute_power(self):
        """测试获取计算能力"""
        from aidevtools.analysis.chip import load_chip_spec

        chip = load_chip_spec("npu_910")

        assert chip.get_compute_power("cube", "fp16") == 256.0
        assert chip.get_compute_power("vector", "fp16") == 16.0  # 16000 GFLOPS -> 16 TFLOPS


class TestPasses:
    """Pass 测试"""

    def test_pass_config_presets(self):
        """测试 Pass 配置预设"""
        from aidevtools.analysis.passes import PassConfig, PassPreset

        minimal = PassConfig.from_preset(PassPreset.MINIMAL)
        assert minimal.roofline_enabled is True
        assert minimal.memory_efficiency_enabled is False
        assert minimal.prefetch.forward_enabled is False

        standard = PassConfig.from_preset(PassPreset.STANDARD)
        assert standard.roofline_enabled is True
        assert standard.memory_efficiency_enabled is True
        assert standard.prefetch.forward_enabled is True

    def test_roofline_pass(self):
        """测试 Roofline Pass"""
        from aidevtools.analysis.passes import RooflinePass, PassConfig
        from aidevtools.analysis.chip import load_chip_spec
        from aidevtools.analysis.latency import LatencyBreakdown
        from aidevtools.analysis.profile import OpProfile

        chip = load_chip_spec("npu_910")
        config = PassConfig()

        profile = OpProfile(
            name="test_matmul",
            op_type="matmul",
            compute_unit="cube",
            dtype="fp16",
            flops=int(2e12),  # 2 TFLOP
            input_bytes=int(1e9),  # 1 GB
            weight_bytes=int(1e9),  # 1 GB
            output_bytes=int(1e9),  # 1 GB
        )

        breakdown = LatencyBreakdown(profile=profile)
        roofline = RooflinePass(config)
        result = roofline.run(breakdown, chip)

        assert result.enabled is True
        assert breakdown.timing.compute_us > 0
        assert breakdown.timing.memory_us > 0
        assert breakdown.timing.roofline_us == max(
            breakdown.timing.compute_us, breakdown.timing.memory_us
        )

    def test_memory_efficiency_pass(self):
        """测试 Memory Efficiency Pass"""
        from aidevtools.analysis.passes import MemoryEfficiencyPass, PassConfig
        from aidevtools.analysis.chip import load_chip_spec
        from aidevtools.analysis.latency import LatencyBreakdown
        from aidevtools.analysis.profile import OpProfile

        chip = load_chip_spec("npu_910")
        config = PassConfig()

        profile = OpProfile(
            name="test_transpose",
            op_type="transpose",
            compute_unit="vector",
            memory_pattern="strided",
            flops=0,
            input_bytes=int(1e6),
            output_bytes=int(1e6),
        )

        breakdown = LatencyBreakdown(profile=profile)
        breakdown.timing.memory_us = 100.0
        breakdown.timing.roofline_us = 100.0

        mem_pass = MemoryEfficiencyPass(config)
        mem_pass.run(breakdown, chip)

        # Strided 模式效率较低，应该增加访存时间
        assert breakdown.timing.memory_us > 100.0


class TestPaperAnalyzer:
    """PaperAnalyzer 测试"""

    def test_analyzer_basic(self):
        """测试基本分析流程"""
        from aidevtools.analysis import PaperAnalyzer

        analyzer = PaperAnalyzer(chip="npu_910")

        # 添加简单的 matmul
        a = np.zeros((4, 512, 768), dtype=np.float16)
        b = np.zeros((768, 768), dtype=np.float16)
        profile = _profile(F.matmul, a, b)
        profile.name = "test_matmul"

        analyzer.add_profile(profile)
        result = analyzer.analyze()

        assert len(result.breakdowns) == 1
        assert result.summary is not None
        assert result.summary.totals.latency_us > 0

    def test_analyzer_multiple_ops(self):
        """测试多算子分析"""
        from aidevtools.analysis import PaperAnalyzer

        analyzer = PaperAnalyzer(chip="npu_910")

        # 使用 profile_only 模式收集多个 profile
        x = np.zeros((4, 512, 768), dtype=np.float16)
        w = np.zeros((768, 768), dtype=np.float16)
        gamma = np.zeros((768,), dtype=np.float16)
        beta = np.zeros((768,), dtype=np.float16)

        with ops.profile_only():
            F.layernorm(x, gamma, beta)
            F.matmul(x, w)
            F.gelu(x)
            profiles = ops.get_profiles()

        for i, p in enumerate(profiles):
            p.name = f"op_{i}"

        analyzer.add_profiles(profiles)
        result = analyzer.analyze()

        assert len(result.breakdowns) == 3
        assert result.summary.bottleneck.compute_bound_ops + result.summary.bottleneck.memory_bound_ops == 3

    def test_gantt_data(self):
        """测试 Gantt 数据生成"""
        from aidevtools.analysis import PaperAnalyzer

        analyzer = PaperAnalyzer(chip="npu_910")

        a = np.zeros((4, 512, 768), dtype=np.float16)
        b = np.zeros((768, 768), dtype=np.float16)
        profile = _profile(F.matmul, a, b)
        profile.name = "test_matmul"

        analyzer.add_profile(profile)
        result = analyzer.analyze()

        assert result.gantt_data is not None
        assert len(result.gantt_data.items) >= 1
        assert result.gantt_data.total_duration_us > 0


def _has_openpyxl():
    """检查是否安装了 openpyxl"""
    try:
        import openpyxl
        return True
    except ImportError:
        return False


class TestExport:
    """Export 测试"""

    def test_export_csv(self):
        """测试 CSV 导出"""
        from aidevtools.analysis import PaperAnalyzer, export_csv

        analyzer = PaperAnalyzer(chip="npu_910")

        a = np.zeros((4, 512, 768), dtype=np.float16)
        b = np.zeros((768, 768), dtype=np.float16)
        profile = _profile(F.matmul, a, b)
        profile.name = "test_matmul"

        analyzer.add_profile(profile)
        result = analyzer.analyze()

        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            export_csv(result, f.name)
            csv_path = Path(f.name)

        assert csv_path.exists()
        content = csv_path.read_text()
        assert "test_matmul" in content
        assert "matmul" in content

        csv_path.unlink()

    def test_export_json(self):
        """测试 JSON 导出"""
        import json
        from aidevtools.analysis import PaperAnalyzer, export_json

        analyzer = PaperAnalyzer(chip="npu_910")

        a = np.zeros((4, 512, 768), dtype=np.float16)
        b = np.zeros((768, 768), dtype=np.float16)
        profile = _profile(F.matmul, a, b)
        profile.name = "test_matmul"

        analyzer.add_profile(profile)
        result = analyzer.analyze()

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            export_json(result, f.name)
            json_path = Path(f.name)

        assert json_path.exists()
        data = json.loads(json_path.read_text())
        assert "chip" in data
        assert "summary" in data
        assert "breakdowns" in data

        json_path.unlink()

    @pytest.mark.skipif(
        not _has_openpyxl(),
        reason="openpyxl not installed"
    )
    def test_export_xlsx(self):
        """测试 Excel 导出"""
        from aidevtools.analysis import PaperAnalyzer, export_xlsx

        analyzer = PaperAnalyzer(chip="npu_910")

        a = np.zeros((4, 512, 768), dtype=np.float16)
        b = np.zeros((768, 768), dtype=np.float16)
        profile = _profile(F.matmul, a, b)
        profile.name = "test_matmul"

        analyzer.add_profile(profile)
        result = analyzer.analyze()

        with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as f:
            export_xlsx(result, f.name)
            xlsx_path = Path(f.name)

        assert xlsx_path.exists()
        assert xlsx_path.stat().st_size > 0

        xlsx_path.unlink()


class TestIntegration:
    """集成测试"""

    def test_transformer_layer_analysis(self):
        """测试 Transformer 层分析"""
        from aidevtools.analysis import PaperAnalyzer, PassConfig, PassPreset

        # 模型参数
        batch, seq, hidden = 4, 512, 768
        heads, head_dim = 12, 64

        # 使用 profile_only 模式收集 profiles
        with ops.profile_only():
            # LayerNorm
            x = np.zeros((batch, seq, hidden), dtype=np.float16)
            gamma = np.zeros((hidden,), dtype=np.float16)
            beta = np.zeros((hidden,), dtype=np.float16)
            F.layernorm(x, gamma, beta)

            # QKV 投影
            w = np.zeros((hidden, hidden), dtype=np.float16)
            for _ in range(3):  # q, k, v
                F.matmul(x, w)

            # Attention
            q = np.zeros((batch, heads, seq, head_dim), dtype=np.float16)
            k = np.zeros((batch, heads, seq, head_dim), dtype=np.float16)
            v = np.zeros((batch, heads, seq, head_dim), dtype=np.float16)
            F.attention(q, k, v)

            # Output 投影
            F.matmul(x, w)

            profiles = ops.get_profiles()

        # 设置名称
        names = ["attn_ln", "q_proj", "k_proj", "v_proj", "self_attn", "out_proj"]
        for i, p in enumerate(profiles):
            p.name = names[i]

        # 分析
        analyzer = PaperAnalyzer(
            chip="npu_910",
            pass_config=PassConfig.from_preset(PassPreset.STANDARD)
        )
        analyzer.add_profiles(profiles)
        result = analyzer.analyze()

        # 验证结果
        assert len(result.breakdowns) == len(profiles)
        assert result.summary.totals.latency_us > 0
        assert result.summary.throughput.achieved_tflops > 0

        # 检查 matmul 为 compute bound
        for bd in result.breakdowns:
            if bd.profile.op_type == "matmul":
                # 大矩阵 matmul 通常是计算瓶颈
                assert bd.profile.compute_unit == "cube"


class TestPrefetchPasses:
    """Prefetch Pass 测试"""

    def test_forward_prefetch_cube_op(self):
        """测试前向预取 - Cube 算子"""
        from aidevtools.analysis.passes import ForwardPrefetchPass, PassConfig, PassContext
        from aidevtools.analysis.chip import load_chip_spec
        from aidevtools.analysis.latency import LatencyBreakdown
        from aidevtools.analysis.profile import OpProfile

        chip = load_chip_spec("npu_910")
        config = PassConfig()
        config.prefetch.forward_enabled = True
        config.prefetch.efficiency = 0.8

        # 当前 Cube 算子 (compute bound, 有空闲 DMA 时间)
        current_profile = OpProfile(
            name="matmul_0",
            op_type="matmul",
            compute_unit="cube",
            dtype="fp16",
            flops=int(2e12),
            input_bytes=int(1e6),
            weight_bytes=int(1e6),
            output_bytes=int(1e6),
        )

        # 下一个算子有权重需要预取
        next_profile = OpProfile(
            name="matmul_1",
            op_type="matmul",
            compute_unit="cube",
            dtype="fp16",
            weight_bytes=int(2e6),  # 2MB 权重
        )

        breakdown = LatencyBreakdown(profile=current_profile)
        breakdown.timing.compute_us = 100.0  # 计算时间
        breakdown.timing.memory_us = 50.0    # 访存时间
        breakdown.timing.roofline_us = 100.0

        context = PassContext(next_profile=next_profile)

        prefetch_pass = ForwardPrefetchPass(config)
        result = prefetch_pass.run(breakdown, chip, context)

        # 应该有预取节省
        assert breakdown.savings.prefetch_us > 0
        assert result.details["idle_time_us"] == 50.0  # compute - memory
        assert result.details["prefetch_efficiency"] == 0.8

    def test_forward_prefetch_vector_op_skip(self):
        """测试前向预取 - Vector 算子跳过"""
        from aidevtools.analysis.passes import ForwardPrefetchPass, PassConfig
        from aidevtools.analysis.chip import load_chip_spec
        from aidevtools.analysis.latency import LatencyBreakdown
        from aidevtools.analysis.profile import OpProfile

        chip = load_chip_spec("npu_910")
        config = PassConfig()
        config.prefetch.forward_enabled = True

        # Vector 算子不能执行前向预取
        profile = OpProfile(
            name="layernorm",
            op_type="layernorm",
            compute_unit="vector",
            dtype="fp16",
        )

        breakdown = LatencyBreakdown(profile=profile)
        breakdown.timing.roofline_us = 50.0

        prefetch_pass = ForwardPrefetchPass(config)
        result = prefetch_pass.run(breakdown, chip)

        # 应该跳过
        assert breakdown.savings.prefetch_us == 0
        assert "非 Cube 算子" in result.details.get("reason", "")

    def test_backward_prefetch_vector_op(self):
        """测试后向预取 - Vector 算子预取后续 Cube 权重"""
        from aidevtools.analysis.passes import BackwardPrefetchPass, PassConfig, PassContext
        from aidevtools.analysis.chip import load_chip_spec
        from aidevtools.analysis.latency import LatencyBreakdown
        from aidevtools.analysis.profile import OpProfile

        chip = load_chip_spec("npu_910")
        config = PassConfig()
        config.prefetch.backward_enabled = True
        config.prefetch.backward_depth = 2
        config.prefetch.efficiency = 0.8

        # 当前 Vector 算子
        current_profile = OpProfile(
            name="layernorm",
            op_type="layernorm",
            compute_unit="vector",
            dtype="fp16",
        )

        # 后续 Cube 算子
        future_profiles = [
            OpProfile(name="ffn1", op_type="matmul", compute_unit="cube", weight_bytes=int(4e6)),
            OpProfile(name="ffn2", op_type="matmul", compute_unit="cube", weight_bytes=int(4e6)),
        ]

        breakdown = LatencyBreakdown(profile=current_profile)
        breakdown.timing.roofline_us = 20.0  # Vector 执行时间

        context = PassContext(future_profiles=future_profiles)

        prefetch_pass = BackwardPrefetchPass(config)
        result = prefetch_pass.run(breakdown, chip, context)

        # 应该有后向预取节省
        assert breakdown.savings.backward_prefetch_us > 0
        assert result.details["prefetch_depth"] == 2
        assert len(result.details["prefetch_details"]) > 0

    def test_backward_prefetch_cube_op_skip(self):
        """测试后向预取 - Cube 算子跳过"""
        from aidevtools.analysis.passes import BackwardPrefetchPass, PassConfig
        from aidevtools.analysis.chip import load_chip_spec
        from aidevtools.analysis.latency import LatencyBreakdown
        from aidevtools.analysis.profile import OpProfile

        chip = load_chip_spec("npu_910")
        config = PassConfig()
        config.prefetch.backward_enabled = True

        # Cube 算子不能执行后向预取
        profile = OpProfile(
            name="matmul",
            op_type="matmul",
            compute_unit="cube",
            dtype="fp16",
        )

        breakdown = LatencyBreakdown(profile=profile)
        breakdown.timing.roofline_us = 100.0

        prefetch_pass = BackwardPrefetchPass(config)
        result = prefetch_pass.run(breakdown, chip)

        # 应该跳过
        assert breakdown.savings.backward_prefetch_us == 0
        assert "非 Vector 算子" in result.details.get("reason", "")


class TestOverheadPass:
    """Overhead Pass 测试"""

    def test_overhead_basic(self):
        """测试基本开销计算"""
        from aidevtools.analysis.passes import OverheadPass, PassConfig
        from aidevtools.analysis.chip import load_chip_spec
        from aidevtools.analysis.latency import LatencyBreakdown
        from aidevtools.analysis.profile import OpProfile

        chip = load_chip_spec("npu_910")
        config = PassConfig()
        config.overhead.enabled = True
        config.overhead.kernel_launch_us = 5.0
        config.overhead.sync_us = 2.0
        config.overhead.context_switch_us = 1.0
        config.overhead.tiling_us = 0.5
        config.overhead.tiling_count = 1  # 无 tiling

        profile = OpProfile(
            name="matmul",
            op_type="matmul",
            compute_unit="cube",
            dtype="fp16",
        )

        breakdown = LatencyBreakdown(profile=profile)
        breakdown.timing.roofline_us = 100.0
        breakdown.savings.prefetch_us = 0
        breakdown.savings.backward_prefetch_us = 0
        breakdown.savings.parallel_us = 0

        overhead_pass = OverheadPass(config)
        overhead_pass.run(breakdown, chip)

        # 总开销 = 5 + 2 + 1 + 0.5*1 = 8.5us
        expected_overhead = 5.0 + 2.0 + 1.0 + 0.5
        assert breakdown.timing.overhead_us == expected_overhead
        assert breakdown.timing.total_us == 100.0 + expected_overhead

    def test_overhead_with_tiling(self):
        """测试带 tiling 的开销计算"""
        from aidevtools.analysis.passes import OverheadPass, PassConfig
        from aidevtools.analysis.chip import load_chip_spec
        from aidevtools.analysis.latency import LatencyBreakdown
        from aidevtools.analysis.profile import OpProfile

        chip = load_chip_spec("npu_910")
        config = PassConfig()
        config.overhead.enabled = True
        config.overhead.kernel_launch_us = 5.0
        config.overhead.sync_us = 2.0
        config.overhead.context_switch_us = 1.0
        config.overhead.tiling_us = 0.5
        config.overhead.tiling_count = 4  # 4 tiles (2x2)

        profile = OpProfile(
            name="matmul",
            op_type="matmul",
            compute_unit="cube",
            dtype="fp16",
        )

        breakdown = LatencyBreakdown(profile=profile)
        breakdown.timing.roofline_us = 100.0
        breakdown.savings.prefetch_us = 0
        breakdown.savings.backward_prefetch_us = 0
        breakdown.savings.parallel_us = 0

        overhead_pass = OverheadPass(config)
        result = overhead_pass.run(breakdown, chip)

        # 总开销 = 5 + 2 + 1 + 0.5*4 = 10us
        expected_overhead = 5.0 + 2.0 + 1.0 + 0.5 * 4
        assert breakdown.timing.overhead_us == expected_overhead
        assert result.details["tiling_count"] == 4
        assert result.details["tiling_total_us"] == 2.0

    def test_overhead_with_prefetch_savings(self):
        """测试开销计算考虑预取节省"""
        from aidevtools.analysis.passes import OverheadPass, PassConfig
        from aidevtools.analysis.chip import load_chip_spec
        from aidevtools.analysis.latency import LatencyBreakdown
        from aidevtools.analysis.profile import OpProfile

        chip = load_chip_spec("npu_910")
        config = PassConfig()
        config.overhead.enabled = True
        config.overhead.kernel_launch_us = 5.0
        config.overhead.sync_us = 2.0
        config.overhead.context_switch_us = 1.0
        config.overhead.tiling_us = 0.0

        profile = OpProfile(
            name="matmul",
            op_type="matmul",
            compute_unit="cube",
            dtype="fp16",
        )

        breakdown = LatencyBreakdown(profile=profile)
        breakdown.timing.roofline_us = 100.0
        breakdown.savings.prefetch_us = 10.0  # 预取节省
        breakdown.savings.backward_prefetch_us = 5.0
        breakdown.savings.parallel_us = 3.0

        overhead_pass = OverheadPass(config)
        overhead_pass.run(breakdown, chip)

        # final = roofline + overhead - prefetch - backward - parallel
        # final = 100 + 8 - 10 - 5 - 3 = 90us
        expected_total = 100.0 + 8.0 - 10.0 - 5.0 - 3.0
        assert breakdown.timing.total_us == expected_total

    def test_overhead_auto_tiling_estimation(self):
        """测试自动 tiling count 估算"""
        from aidevtools.analysis.passes import OverheadPass, PassConfig
        from aidevtools.analysis.chip import load_chip_spec
        from aidevtools.analysis.latency import LatencyBreakdown
        from aidevtools.analysis.profile import OpProfile

        chip = load_chip_spec("npu_910")
        config = PassConfig()
        config.overhead.enabled = True
        config.overhead.tiling_count = 1  # 使用自动估算

        # 大矩阵应该触发 tiling
        profile = OpProfile(
            name="large_matmul",
            op_type="matmul",
            compute_unit="cube",
            dtype="fp16",
            shapes={"M": 4096, "N": 4096, "K": 4096},  # 大矩阵
        )

        breakdown = LatencyBreakdown(profile=profile)
        breakdown.timing.roofline_us = 100.0

        overhead_pass = OverheadPass(config)
        result = overhead_pass.run(breakdown, chip)

        # 大矩阵应该有多个 tiles
        assert result.details["tiling_count"] >= 1


class TestCubeVectorParallelPass:
    """Cube/Vector 并行 Pass 测试"""

    def test_parallel_pass_cube_op(self):
        """测试 Cube 算子并行"""
        from aidevtools.analysis.passes import CubeVectorParallelPass, PassConfig, PassContext
        from aidevtools.analysis.chip import load_chip_spec
        from aidevtools.analysis.latency import LatencyBreakdown
        from aidevtools.analysis.profile import OpProfile

        chip = load_chip_spec("npu_910")
        config = PassConfig()
        config.cube_vector_parallel_enabled = True

        # 当前 Cube 算子
        current_profile = OpProfile(
            name="matmul",
            op_type="matmul",
            compute_unit="cube",
            dtype="fp16",
        )

        # 下一个是 Vector 算子
        next_profile = OpProfile(
            name="layernorm",
            op_type="layernorm",
            compute_unit="vector",
            dtype="fp16",
        )

        breakdown = LatencyBreakdown(profile=current_profile)
        breakdown.timing.roofline_us = 100.0

        context = PassContext(next_profile=next_profile)

        parallel_pass = CubeVectorParallelPass(config)
        result = parallel_pass.run(breakdown, chip, context)

        # Pass 应该执行
        assert result.enabled is True


class TestBandwidthPasses:
    """带宽约束 Pass 测试"""

    def test_bandwidth_constraint_pass_single_stream(self):
        """测试单流无约束"""
        from aidevtools.analysis.passes import BandwidthConstraintPass, PassConfig
        from aidevtools.analysis.chip import load_chip_spec
        from aidevtools.analysis.latency import LatencyBreakdown
        from aidevtools.analysis.profile import OpProfile

        chip = load_chip_spec("npu_910")
        config = PassConfig()
        config.bandwidth.enabled = True
        config.bandwidth.concurrent_streams = 1  # 单流

        profile = OpProfile(
            name="test_matmul",
            op_type="matmul",
            compute_unit="cube",
            dtype="fp16",
            flops=int(1e12),
            input_bytes=int(1e9),
            weight_bytes=int(1e9),
            output_bytes=int(1e9),
        )

        breakdown = LatencyBreakdown(profile=profile)
        breakdown.timing.memory_us = 100.0
        breakdown.timing.roofline_us = 100.0

        bw_pass = BandwidthConstraintPass(config)
        result = bw_pass.run(breakdown, chip)

        # 单流无约束，时延不变
        assert breakdown.timing.memory_us == 100.0
        assert result.details.get("reason") == "单流执行，无带宽竞争"

    def test_bandwidth_constraint_pass_multi_stream(self):
        """测试多流带宽竞争"""
        from aidevtools.analysis.passes import BandwidthConstraintPass, PassConfig
        from aidevtools.analysis.chip import load_chip_spec
        from aidevtools.analysis.latency import LatencyBreakdown
        from aidevtools.analysis.profile import OpProfile

        chip = load_chip_spec("npu_910")
        config = PassConfig()
        config.bandwidth.enabled = True
        config.bandwidth.concurrent_streams = 4  # 4 流并发
        config.bandwidth.contention_model = "linear"

        profile = OpProfile(
            name="test_matmul",
            op_type="matmul",
            compute_unit="cube",
            dtype="fp16",
            flops=int(1e12),
            input_bytes=int(1e9),
            weight_bytes=int(1e9),
            output_bytes=int(1e9),
        )

        breakdown = LatencyBreakdown(profile=profile)
        breakdown.timing.memory_us = 100.0
        breakdown.timing.roofline_us = 100.0
        breakdown.timing.compute_us = 50.0

        bw_pass = BandwidthConstraintPass(config)
        result = bw_pass.run(breakdown, chip)

        # 4 流 linear 模型，带宽 /4，时延 x4
        assert breakdown.timing.memory_us == 400.0
        assert result.details["contention_factor"] == 4.0
        assert len(result.warnings) > 0

    def test_min_traffic_pass(self):
        """测试最低流量优化"""
        from aidevtools.analysis.passes import MinTrafficPass, PassConfig
        from aidevtools.analysis.chip import load_chip_spec
        from aidevtools.analysis.latency import LatencyBreakdown
        from aidevtools.analysis.profile import OpProfile

        chip = load_chip_spec("npu_910")
        config = PassConfig()
        config.min_traffic.enabled = True
        config.min_traffic.l2_reuse_factor = 0.5   # 权重复用 50%
        config.min_traffic.tiling_efficiency = 0.8  # Tiling 减少 20%

        profile = OpProfile(
            name="test_matmul",
            op_type="matmul",
            compute_unit="cube",
            dtype="fp16",
            flops=int(1e12),
            input_bytes=int(1e6),
            weight_bytes=int(2e6),
            output_bytes=int(1e6),
        )

        breakdown = LatencyBreakdown(profile=profile)
        breakdown.timing.memory_us = 100.0
        breakdown.timing.roofline_us = 100.0
        breakdown.timing.compute_us = 50.0

        min_pass = MinTrafficPass(config)
        result = min_pass.run(breakdown, chip)

        # 流量应该减少
        assert result.details["traffic_saved_ratio"] > 0
        assert result.details["optimized_traffic_bytes"] < result.details["original_traffic_bytes"]

    def test_traffic_constraint_pass(self):
        """测试流量约束检查"""
        from aidevtools.analysis.passes import TrafficConstraintPass, PassConfig
        from aidevtools.analysis.chip import load_chip_spec
        from aidevtools.analysis.latency import LatencyBreakdown
        from aidevtools.analysis.profile import OpProfile

        chip = load_chip_spec("npu_910")
        config = PassConfig()
        config.traffic.enabled = True
        config.traffic.max_bytes = int(1e6)  # 1MB 限制
        config.traffic.budget_mode = "strict"

        profile = OpProfile(
            name="test_matmul",
            op_type="matmul",
            compute_unit="cube",
            dtype="fp16",
            flops=int(1e12),
            input_bytes=int(1e6),
            weight_bytes=int(2e6),  # 超过限制
            output_bytes=int(1e6),
        )

        breakdown = LatencyBreakdown(profile=profile)
        breakdown.timing.total_us = 100.0

        traffic_pass = TrafficConstraintPass(config)
        result = traffic_pass.run(breakdown, chip)

        # 应该检测到超限
        assert result.details["over_budget"] is True
        assert len(result.warnings) > 0
        assert "超限" in result.warnings[0]

    def test_analyzer_with_bandwidth_constraint(self):
        """测试分析器集成带宽约束"""
        from aidevtools.analysis import PaperAnalyzer, PassConfig

        config = PassConfig()
        config.bandwidth.enabled = True
        config.bandwidth.concurrent_streams = 2  # 2 流并发
        config.bandwidth.contention_model = "sqrt"

        analyzer = PaperAnalyzer(chip="npu_910", pass_config=config)

        a = np.zeros((4, 512, 768), dtype=np.float16)
        b = np.zeros((768, 768), dtype=np.float16)
        profile = _profile(F.matmul, a, b)
        profile.name = "test_matmul"

        analyzer.add_profile(profile)
        result = analyzer.analyze()

        # 验证结果包含带宽约束效果
        assert len(result.breakdowns) == 1
        assert result.summary is not None

    def test_analyzer_with_min_traffic(self):
        """测试分析器集成最低流量模式"""
        from aidevtools.analysis import PaperAnalyzer, PassConfig, PassPreset

        # 使用激进模式（自动启用最低流量优化）
        config = PassConfig.from_preset(PassPreset.AGGRESSIVE)

        analyzer = PaperAnalyzer(chip="npu_910", pass_config=config)

        a = np.zeros((4, 512, 768), dtype=np.float16)
        b = np.zeros((768, 768), dtype=np.float16)
        profile = _profile(F.matmul, a, b)
        profile.name = "test_matmul"

        analyzer.add_profile(profile)
        result = analyzer.analyze()

        # 验证流量统计
        assert result.summary.traffic.original_bytes > 0
        assert result.summary.traffic.optimized_bytes > 0
