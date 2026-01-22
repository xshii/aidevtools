"""
Paper Analysis 模块单元测试
"""

import pytest
import numpy as np
from pathlib import Path
import tempfile


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
        from aidevtools.analysis.profile import profile_matmul, OpProfile

        a = np.zeros((4, 512, 768), dtype=np.float16)
        b = np.zeros((768, 768), dtype=np.float16)
        profile = profile_matmul(a, b)

        assert profile.op_type == "matmul"
        assert profile.compute_unit == "cube"
        # FLOPs = 2 * batch * M * K * N = 2 * 4 * 512 * 768 * 768
        expected_flops = 2 * 4 * 512 * 768 * 768
        assert profile.flops == expected_flops
        # Input: 4 * 512 * 768 * 2 bytes
        assert profile.input_bytes == 4 * 512 * 768 * 2
        # Weight: 768 * 768 * 2 bytes
        assert profile.weight_bytes == 768 * 768 * 2
        # Output: 4 * 512 * 768 * 2 bytes
        assert profile.output_bytes == 4 * 512 * 768 * 2

    def test_layernorm_profile(self):
        """测试 LayerNorm profile"""
        from aidevtools.analysis.profile import profile_layernorm

        x = np.zeros((4, 512, 768), dtype=np.float16)
        gamma = np.zeros((768,), dtype=np.float16)
        beta = np.zeros((768,), dtype=np.float16)
        profile = profile_layernorm(x, gamma, beta)

        assert profile.op_type == "layernorm"
        assert profile.compute_unit == "vector"
        # FLOPs: ~8 ops/element
        assert profile.flops == 8 * x.size

    def test_attention_profile(self):
        """测试 Attention profile"""
        from aidevtools.analysis.profile import profile_attention

        batch, heads, seq, head_dim = 4, 12, 512, 64
        q = np.zeros((batch, heads, seq, head_dim), dtype=np.float16)
        k = np.zeros((batch, heads, seq, head_dim), dtype=np.float16)
        v = np.zeros((batch, heads, seq, head_dim), dtype=np.float16)
        profile = profile_attention(q, k, v)

        assert profile.op_type == "attention"
        assert profile.compute_unit == "cube"
        assert profile.flops > 0

    def test_transpose_profile(self):
        """测试 Transpose profile"""
        from aidevtools.analysis.profile import profile_transpose

        x = np.zeros((4, 12, 512, 64), dtype=np.float16)
        profile = profile_transpose(x, axes=(0, 2, 1, 3))

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
        from aidevtools.analysis.chip import load_chip_spec, list_chips

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
        assert minimal.forward_prefetch_enabled is False

        standard = PassConfig.from_preset(PassPreset.STANDARD)
        assert standard.roofline_enabled is True
        assert standard.memory_efficiency_enabled is True
        assert standard.forward_prefetch_enabled is True

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
        assert breakdown.compute_time_us > 0
        assert breakdown.memory_time_us > 0
        assert breakdown.roofline_time_us == max(
            breakdown.compute_time_us, breakdown.memory_time_us
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
        breakdown.memory_time_us = 100.0
        breakdown.roofline_time_us = 100.0

        mem_pass = MemoryEfficiencyPass(config)
        result = mem_pass.run(breakdown, chip)

        # Strided 模式效率较低，应该增加访存时间
        assert breakdown.memory_time_us > 100.0


class TestPaperAnalyzer:
    """PaperAnalyzer 测试"""

    def test_analyzer_basic(self):
        """测试基本分析流程"""
        from aidevtools.analysis import PaperAnalyzer
        from aidevtools.analysis.profile import profile_matmul

        analyzer = PaperAnalyzer(chip="npu_910")

        # 添加简单的 matmul
        a = np.zeros((4, 512, 768), dtype=np.float16)
        b = np.zeros((768, 768), dtype=np.float16)
        profile = profile_matmul(a, b)
        profile.name = "test_matmul"

        analyzer.add_profile(profile)
        result = analyzer.analyze()

        assert len(result.breakdowns) == 1
        assert result.summary is not None
        assert result.summary.total_latency_us > 0

    def test_analyzer_multiple_ops(self):
        """测试多算子分析"""
        from aidevtools.analysis import PaperAnalyzer
        from aidevtools.analysis.profile import (
            profile_matmul,
            profile_layernorm,
            profile_gelu,
        )

        analyzer = PaperAnalyzer(chip="npu_910")

        # 添加多个算子
        x = np.zeros((4, 512, 768), dtype=np.float16)
        w = np.zeros((768, 768), dtype=np.float16)
        gamma = np.zeros((768,), dtype=np.float16)
        beta = np.zeros((768,), dtype=np.float16)

        profiles = [
            profile_layernorm(x, gamma, beta),
            profile_matmul(x, w),
            profile_gelu(x),
        ]
        for i, p in enumerate(profiles):
            p.name = f"op_{i}"

        analyzer.add_profiles(profiles)
        result = analyzer.analyze()

        assert len(result.breakdowns) == 3
        assert result.summary.compute_bound_ops + result.summary.memory_bound_ops == 3

    def test_gantt_data(self):
        """测试 Gantt 数据生成"""
        from aidevtools.analysis import PaperAnalyzer
        from aidevtools.analysis.profile import profile_matmul

        analyzer = PaperAnalyzer(chip="npu_910")

        a = np.zeros((4, 512, 768), dtype=np.float16)
        b = np.zeros((768, 768), dtype=np.float16)
        profile = profile_matmul(a, b)
        profile.name = "test_matmul"

        analyzer.add_profile(profile)
        result = analyzer.analyze()

        assert result.gantt_data is not None
        assert len(result.gantt_data.items) >= 1
        assert result.gantt_data.total_time_us > 0


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
        from aidevtools.analysis.profile import profile_matmul

        analyzer = PaperAnalyzer(chip="npu_910")

        a = np.zeros((4, 512, 768), dtype=np.float16)
        b = np.zeros((768, 768), dtype=np.float16)
        profile = profile_matmul(a, b)
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
        from aidevtools.analysis.profile import profile_matmul

        analyzer = PaperAnalyzer(chip="npu_910")

        a = np.zeros((4, 512, 768), dtype=np.float16)
        b = np.zeros((768, 768), dtype=np.float16)
        profile = profile_matmul(a, b)
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
        from aidevtools.analysis.profile import profile_matmul

        analyzer = PaperAnalyzer(chip="npu_910")

        a = np.zeros((4, 512, 768), dtype=np.float16)
        b = np.zeros((768, 768), dtype=np.float16)
        profile = profile_matmul(a, b)
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
        from aidevtools.analysis.profile import (
            profile_matmul,
            profile_layernorm,
            profile_attention,
            profile_gelu,
            profile_add,
        )

        # 模型参数
        batch, seq, hidden = 4, 512, 768
        heads, head_dim = 12, 64
        ffn_hidden = 3072

        profiles = []

        # LayerNorm
        x = np.zeros((batch, seq, hidden), dtype=np.float16)
        gamma = np.zeros((hidden,), dtype=np.float16)
        beta = np.zeros((hidden,), dtype=np.float16)
        ln = profile_layernorm(x, gamma, beta)
        ln.name = "attn_ln"
        profiles.append(ln)

        # QKV 投影
        w = np.zeros((hidden, hidden), dtype=np.float16)
        for name in ["q_proj", "k_proj", "v_proj"]:
            p = profile_matmul(x, w)
            p.name = name
            profiles.append(p)

        # Attention
        q = np.zeros((batch, heads, seq, head_dim), dtype=np.float16)
        k = np.zeros((batch, heads, seq, head_dim), dtype=np.float16)
        v = np.zeros((batch, heads, seq, head_dim), dtype=np.float16)
        attn = profile_attention(q, k, v)
        attn.name = "self_attn"
        profiles.append(attn)

        # Output 投影
        out_proj = profile_matmul(x, w)
        out_proj.name = "out_proj"
        profiles.append(out_proj)

        # 分析
        analyzer = PaperAnalyzer(
            chip="npu_910",
            pass_config=PassConfig.from_preset(PassPreset.STANDARD)
        )
        analyzer.add_profiles(profiles)
        result = analyzer.analyze()

        # 验证结果
        assert len(result.breakdowns) == len(profiles)
        assert result.summary.total_latency_us > 0
        assert result.summary.achieved_tflops > 0

        # 检查 matmul 为 compute bound
        for bd in result.breakdowns:
            if bd.profile.op_type == "matmul":
                # 大矩阵 matmul 通常是计算瓶颈
                assert bd.profile.compute_unit == "cube"


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
        config.bandwidth_constraint_enabled = True
        config.concurrent_streams = 1  # 单流

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
        breakdown.memory_time_us = 100.0
        breakdown.roofline_time_us = 100.0

        bw_pass = BandwidthConstraintPass(config)
        result = bw_pass.run(breakdown, chip)

        # 单流无约束，时延不变
        assert breakdown.memory_time_us == 100.0
        assert result.details.get("reason") == "单流执行，无带宽竞争"

    def test_bandwidth_constraint_pass_multi_stream(self):
        """测试多流带宽竞争"""
        from aidevtools.analysis.passes import BandwidthConstraintPass, PassConfig
        from aidevtools.analysis.chip import load_chip_spec
        from aidevtools.analysis.latency import LatencyBreakdown
        from aidevtools.analysis.profile import OpProfile

        chip = load_chip_spec("npu_910")
        config = PassConfig()
        config.bandwidth_constraint_enabled = True
        config.concurrent_streams = 4  # 4 流并发
        config.bandwidth_contention_model = "linear"

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
        breakdown.memory_time_us = 100.0
        breakdown.roofline_time_us = 100.0
        breakdown.compute_time_us = 50.0

        bw_pass = BandwidthConstraintPass(config)
        result = bw_pass.run(breakdown, chip)

        # 4 流 linear 模型，带宽 /4，时延 x4
        assert breakdown.memory_time_us == 400.0
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
        config.min_traffic_mode_enabled = True
        config.l2_reuse_factor = 0.5   # 权重复用 50%
        config.tiling_efficiency = 0.8  # Tiling 减少 20%

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
        breakdown.memory_time_us = 100.0
        breakdown.roofline_time_us = 100.0
        breakdown.compute_time_us = 50.0

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
        config.traffic_constraint_enabled = True
        config.max_traffic_bytes = int(1e6)  # 1MB 限制
        config.traffic_budget_mode = "strict"

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
        breakdown.total_time_us = 100.0

        traffic_pass = TrafficConstraintPass(config)
        result = traffic_pass.run(breakdown, chip)

        # 应该检测到超限
        assert result.details["over_budget"] is True
        assert len(result.warnings) > 0
        assert "超限" in result.warnings[0]

    def test_analyzer_with_bandwidth_constraint(self):
        """测试分析器集成带宽约束"""
        from aidevtools.analysis import PaperAnalyzer, PassConfig
        from aidevtools.analysis.profile import profile_matmul

        config = PassConfig()
        config.bandwidth_constraint_enabled = True
        config.concurrent_streams = 2  # 2 流并发
        config.bandwidth_contention_model = "sqrt"

        analyzer = PaperAnalyzer(chip="npu_910", pass_config=config)

        a = np.zeros((4, 512, 768), dtype=np.float16)
        b = np.zeros((768, 768), dtype=np.float16)
        profile = profile_matmul(a, b)
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
        from aidevtools.analysis.profile import profile_matmul
        profile = profile_matmul(a, b)
        profile.name = "test_matmul"

        analyzer.add_profile(profile)
        result = analyzer.analyze()

        # 验证流量统计
        assert result.summary.total_original_traffic_bytes > 0
        assert result.summary.total_optimized_traffic_bytes > 0
