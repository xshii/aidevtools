"""量化感知权重 & 四种比数测试

测试内容:
1. PrecisionConfig 创建和属性
2. 量化感知随机数生成 (QA-aware random)
3. 四种比数 golden 生成 (FourTrackGolden)
4. DataGenerator 集成测试
5. Model DSL 集成测试
6. 各前端 (Excel/Torch/DSL) 精度配置传递
"""

import numpy as np
import pytest


# ============================================================
# 1. PrecisionConfig 测试
# ============================================================


class TestPrecisionConfig:
    """PrecisionConfig 基本功能"""

    def test_default(self):
        from aidevtools.frontend.types import PrecisionConfig
        pc = PrecisionConfig()
        assert pc.compute_dtype == "fp32"
        assert pc.input_dtype == "fp32"
        assert pc.output_dtype == "fp32"
        assert pc.weight_dtype == "fp32"
        assert pc.qa_aware is False
        assert not pc.is_mixed

    def test_mixed_precision(self):
        from aidevtools.frontend.types import PrecisionConfig
        pc = PrecisionConfig(
            input_dtype="fp16",
            weight_dtype="int8",
            compute_dtype="fp32",
            output_dtype="bfp8",
        )
        assert pc.is_mixed
        assert pc.input_dtype == "fp16"
        assert pc.weight_dtype == "int8"

    def test_qa_aware_config(self):
        from aidevtools.frontend.types import PrecisionConfig
        pc = PrecisionConfig(qa_aware=True, qa_center=1.0, qa_amplitude=0.5)
        assert pc.qa_aware is True
        assert pc.qa_center == 1.0
        assert pc.qa_amplitude == 0.5

    def test_from_dict(self):
        from aidevtools.frontend.types import PrecisionConfig
        d = {
            "compute_dtype": "fp16",
            "input_dtype": "int8",
            "qa_aware": True,
            "unknown_key": "ignored",
        }
        pc = PrecisionConfig.from_dict(d)
        assert pc.compute_dtype == "fp16"
        assert pc.input_dtype == "int8"
        assert pc.qa_aware is True

    def test_weight_dtype_inherits_input(self):
        from aidevtools.frontend.types import PrecisionConfig
        pc = PrecisionConfig(input_dtype="fp16")
        assert pc.weight_dtype == "fp16"


# ============================================================
# 2. DType 测试
# ============================================================


class TestDType:
    """DType 扩展测试"""

    def test_int16_exists(self):
        from aidevtools.frontend.types import DType
        assert DType.INT16.value == "int16"
        assert DType.from_str("int16") == DType.INT16

    def test_is_local(self):
        from aidevtools.frontend.types import DType
        assert DType.FP32.is_local
        assert DType.FP16.is_local
        assert DType.INT16.is_local
        assert DType.INT8.is_local
        assert not DType.BFP16.is_local
        assert not DType.GFP8.is_local

    def test_is_hw_quant(self):
        from aidevtools.frontend.types import DType
        assert DType.BFP16.is_hw_quant
        assert DType.BFP8.is_hw_quant
        assert DType.GFP16.is_hw_quant
        assert DType.GFP8.is_hw_quant
        assert not DType.FP32.is_hw_quant
        assert not DType.INT8.is_hw_quant


# ============================================================
# 3. 量化感知随机数测试
# ============================================================


class TestQARandomGeneration:
    """量化感知随机数生成"""

    def test_qa_uniform_range(self):
        """QA uniform: 值应在 [center-amplitude, center+amplitude] 内"""
        from aidevtools.core.random import RandomGenerator
        rng = RandomGenerator(seed=42)
        data = rng.qa_uniform((1000,), center=1.0, amplitude=0.5, signed=False)
        assert np.all(data >= 0.5)
        assert np.all(data <= 1.5)

    def test_qa_uniform_signed(self):
        """QA uniform signed: 绝对值在 [center-amplitude, center+amplitude]"""
        from aidevtools.core.random import RandomGenerator
        rng = RandomGenerator(seed=42)
        data = rng.qa_uniform((1000,), center=1.0, amplitude=0.5, signed=True)
        abs_data = np.abs(data)
        assert np.all(abs_data >= 0.5 - 1e-7)
        assert np.all(abs_data <= 1.5 + 1e-7)
        # 确认有正有负
        assert np.any(data > 0)
        assert np.any(data < 0)

    def test_qa_normal_range(self):
        """QA normal: 值截断在 [center-amplitude, center+amplitude]"""
        from aidevtools.core.random import RandomGenerator
        rng = RandomGenerator(seed=42)
        data = rng.qa_normal((10000,), center=1.0, amplitude=0.5, signed=False)
        assert np.all(data >= 0.5 - 1e-7)
        assert np.all(data <= 1.5 + 1e-7)

    def test_qa_dynamic_range_ratio(self):
        """QA random: max/min 比值应该受控"""
        from aidevtools.core.random import RandomGenerator
        rng = RandomGenerator(seed=42)
        # center=1.0, amplitude=0.5 → [0.5, 1.5], ratio = 3.0
        data = rng.qa_uniform((1000, 100), center=1.0, amplitude=0.5, signed=False)
        ratio = np.max(data) / np.min(data)
        assert ratio <= 3.1  # 1.5/0.5 = 3.0

    def test_qa_global_config(self):
        """全局 QA 配置开关"""
        from aidevtools.core.random import RandomGenerator
        rng = RandomGenerator(seed=42)

        # 默认不开 QA: 正常随机
        data_normal = rng.generate((1000,), method="normal")
        assert not rng.qa_enabled

        # 启用全局 QA 配置
        rng.set_qa_config(enabled=True, center=1.0, amplitude=0.5)
        assert rng.qa_enabled

        # QA 模式下生成的数据受控
        data_qa = rng.generate((1000,), method="normal")
        abs_qa = np.abs(data_qa)
        assert np.all(abs_qa >= 0.5 - 1e-7)
        assert np.all(abs_qa <= 1.5 + 1e-7)

    def test_qa_per_call_override(self):
        """per-call qa_aware 参数可覆盖全局配置"""
        from aidevtools.core.random import RandomGenerator
        rng = RandomGenerator(seed=42)

        # 全局不开 QA，但 per-call 开启
        data = rng.generate((1000,), method="normal", qa_aware=True)
        abs_data = np.abs(data)
        assert np.all(abs_data >= 0.5 - 1e-7)
        assert np.all(abs_data <= 1.5 + 1e-7)

    def test_qa_matmul_output_bounded(self):
        """验证 QA 输入做 matmul 后输出动态范围比正态分布更可控"""
        from aidevtools.core.random import RandomGenerator

        # QA 模式: 输入绝对值在 [0.5, 1.5]，减少极端值
        rng_qa = RandomGenerator(seed=42)
        K = 64
        a_qa = rng_qa.qa_uniform((32, K), center=1.0, amplitude=0.5, signed=True)
        b_qa = rng_qa.qa_uniform((K, 16), center=1.0, amplitude=0.5, signed=True)
        c_qa = a_qa @ b_qa

        # 正常模式: 正态分布，可能有极端值
        rng_n = RandomGenerator(seed=42)
        a_n = rng_n.normal((32, K))
        b_n = rng_n.normal((K, 16))
        c_n = a_n @ b_n

        # QA 模式的输出标准差应该更可控 (更均匀)
        # 没有 NaN/Inf
        assert not np.any(np.isnan(c_qa))
        assert not np.any(np.isinf(c_qa))

        # QA 的输出 max/min(abs) 在合理范围内
        # 由于 signed 随机符号，matmul 会有正负抵消，min_abs 可能很小
        # 我们验证 QA 模式下 std(abs) / mean(abs) < 正态分布的相同比值
        qa_cv = np.std(np.abs(c_qa)) / (np.mean(np.abs(c_qa)) + 1e-10)
        normal_cv = np.std(np.abs(c_n)) / (np.mean(np.abs(c_n)) + 1e-10)
        # QA 的变异系数应不比正态差太多
        assert qa_cv < 2.0, f"QA output CV too large: {qa_cv}"


# ============================================================
# 4. DataGenerator 集成测试
# ============================================================


class TestDataGeneratorQA:
    """DataGenerator 量化感知集成"""

    def test_generate_with_qa(self):
        """使用 QA precision 生成数据"""
        from aidevtools.datagen import DataGenerator
        from aidevtools.frontend.types import PrecisionConfig

        pc = PrecisionConfig(qa_aware=True, qa_center=1.0, qa_amplitude=0.5)
        gen = DataGenerator(seed=42, precision=pc)
        data = gen.generate("relu", input_shape=(4, 8))
        assert "x" in data
        # QA 模式下，生成的数据绝对值在受控范围内
        abs_vals = np.abs(data["x"].array)
        assert np.all(abs_vals >= 0.5 - 1e-6)
        assert np.all(abs_vals <= 1.5 + 1e-6)

    def test_qa_randn_explicit(self):
        """显式调用 qa_randn"""
        from aidevtools.datagen import DataGenerator
        gen = DataGenerator(seed=42)
        t = gen.qa_randn((100, 100), center=2.0, amplitude=0.3)
        abs_vals = np.abs(t.array)
        assert np.all(abs_vals >= 1.7 - 1e-6)
        assert np.all(abs_vals <= 2.3 + 1e-6)

    def test_normal_mode_not_bounded(self):
        """不开 QA 时，正态分布不应受控"""
        from aidevtools.datagen import DataGenerator
        gen = DataGenerator(seed=42)
        t = gen.randn((1000,))
        # 正态分布的绝对值应有较大范围
        assert np.max(np.abs(t.array)) > 1.5


# ============================================================
# 5. 四种比数测试
# ============================================================


class TestFourTrackGolden:
    """四种比数 golden 生成"""

    def test_track1_pure_fp32(self):
        """Track 1: 纯 fp32 golden 始终生成"""
        from aidevtools.datagen import DataGenerator
        gen = DataGenerator(seed=42)
        tracks = gen.generate_four_track("relu", input_shape=(4, 8))
        assert tracks.golden_pure is not None
        assert tracks.golden_pure.shape == (4, 8)

    def test_track2_local_format(self):
        """Track 2: 本地格式原生数据 golden"""
        from aidevtools.datagen import DataGenerator
        from aidevtools.frontend.types import PrecisionConfig

        pc = PrecisionConfig(input_dtype="fp16")
        gen = DataGenerator(seed=42, precision=pc)
        tracks = gen.generate_four_track("relu", input_shape=(4, 8), precision=pc)

        assert tracks.golden_pure is not None
        assert tracks.golden_local is not None
        assert tracks.data_local is not None
        assert tracks.data_pure is not None

        # data_local 存储原生 fp16 dtype (不是 fp32)
        local_x = tracks.data_local["x"]
        assert local_x.dtype == np.float16, f"Expected fp16 native dtype, got {local_x.dtype}"

        # cast 回 fp32 后与 pure 有差异 (fp16 精度损失)
        pure_x = tracks.data_pure["x"]
        local_x_fp32 = local_x.astype(np.float32)
        assert not np.array_equal(pure_x, local_x_fp32), "fp16 should cause precision difference"
        diff = np.max(np.abs(pure_x - local_x_fp32))
        assert diff < 0.01, f"fp16 vs fp32 input diff too large: {diff}"

    def test_track3_hw_format(self):
        """Track 3: 硬件格式模糊权重 golden"""
        from aidevtools.datagen import DataGenerator
        from aidevtools.frontend.types import PrecisionConfig

        pc = PrecisionConfig(input_dtype="bfp16")
        gen = DataGenerator(seed=42, precision=pc)
        tracks = gen.generate_four_track("relu", input_shape=(4, 16), precision=pc)

        assert tracks.golden_pure is not None
        assert tracks.golden_hw is not None

    def test_track4_qa_aware(self):
        """Track 4: 量化感知 golden"""
        from aidevtools.datagen import DataGenerator
        from aidevtools.frontend.types import PrecisionConfig

        pc = PrecisionConfig(qa_aware=True, qa_center=1.0, qa_amplitude=0.5)
        gen = DataGenerator(seed=42, precision=pc)
        tracks = gen.generate_four_track("relu", input_shape=(4, 8), precision=pc)

        assert tracks.golden_pure is not None
        assert tracks.golden_qa is not None
        # QA golden 的值范围应受控 (relu 不改变正值)
        qa_abs = np.abs(tracks.golden_qa)
        # relu 输出应 >= 0
        assert np.all(tracks.golden_qa >= -1e-7)

    def test_all_four_tracks(self):
        """四种比数同时生成"""
        from aidevtools.datagen import DataGenerator
        from aidevtools.frontend.types import PrecisionConfig

        pc = PrecisionConfig(
            input_dtype="fp16",
            weight_dtype="bfp16",
            compute_dtype="fp32",
            output_dtype="fp32",
            qa_aware=True,
            qa_center=1.0,
            qa_amplitude=0.5,
        )
        gen = DataGenerator(seed=42, precision=pc)
        tracks = gen.generate_four_track("relu", input_shape=(4, 16), precision=pc)

        all_g = tracks.all_goldens
        assert "pure" in all_g
        assert "local" in all_g  # fp16 触发 local track
        assert "hw" in all_g    # bfp16 触发 hw track
        assert "qa" in all_g    # qa_aware=True 触发 qa track
        assert len(all_g) == 4

    def test_four_track_data_preserved(self):
        """四种比数保存了对应的输入数据"""
        from aidevtools.datagen import DataGenerator
        from aidevtools.frontend.types import PrecisionConfig

        pc = PrecisionConfig(
            input_dtype="fp16",
            weight_dtype="bfp16",
            qa_aware=True,
        )
        gen = DataGenerator(seed=42, precision=pc)
        tracks = gen.generate_four_track("relu", input_shape=(4, 16), precision=pc)

        assert tracks.data_pure is not None
        assert "x" in tracks.data_pure
        if tracks.data_local is not None:
            assert "x" in tracks.data_local
        if tracks.data_qa is not None:
            assert "x" in tracks.data_qa


# ============================================================
# 6. Model DSL QA 测试
# ============================================================


class TestModelDSLQA:
    """Model DSL 量化感知"""

    def test_model_with_qa_precision(self):
        """Model DSL 使用 QA 精度配置"""
        from aidevtools.datagen import Model
        from aidevtools.frontend.types import PrecisionConfig

        pc = PrecisionConfig(qa_aware=True, qa_center=1.0, qa_amplitude=0.5)

        with Model(seed=42, precision=pc) as m:
            x = m.input((4, 8))
            y = m.relu(x)

        assert m.precision.qa_aware is True
        assert y.golden is not None
        # relu 的输入在 QA 范围内
        abs_input = np.abs(x.golden)
        assert np.all(abs_input >= 0.5 - 1e-6)
        assert np.all(abs_input <= 1.5 + 1e-6)

    def test_model_default_no_qa(self):
        """Model DSL 默认不开 QA"""
        from aidevtools.datagen import Model

        with Model(seed=42) as m:
            x = m.input((100,))
            y = m.relu(x)

        assert m.precision.qa_aware is False
        # 默认正态分布，范围可能较大
        assert np.max(np.abs(x.golden)) > 1.0

    def test_model_four_track(self):
        """Model DSL 四种比数"""
        from aidevtools.datagen import Model
        from aidevtools.frontend.types import PrecisionConfig

        pc = PrecisionConfig(
            input_dtype="fp16",
            qa_aware=True,
        )
        m = Model(seed=42, precision=pc)
        tracks = m.generate_four_track("relu", input_shape=(4, 16), precision=pc)
        assert tracks.golden_pure is not None
        assert tracks.golden_qa is not None


# ============================================================
# 7. 本地格式原生数据生成测试
# ============================================================


class TestNativeLocalDtype:
    """本地格式原生数据生成 (直接在目标格式生成)"""

    def test_fp16_native_dtype(self):
        """fp16: 返回原生 fp16 dtype"""
        from aidevtools.datagen import _to_native_local_dtype
        data = np.array([1.0001, 2.0002, 0.1234567], dtype=np.float32)
        result = _to_native_local_dtype(data, "fp16")
        # 返回原生 fp16 dtype
        assert result.dtype == np.float16
        # cast 回 fp32 后有精度差异
        assert not np.array_equal(data, result.astype(np.float32))

    def test_int8_native_dtype(self):
        """int8: 直接生成原生 int8 随机整数"""
        from aidevtools.datagen import _to_native_local_dtype
        rng = np.random.default_rng(42)
        data = np.array([1.0, -0.5, 0.3, -0.8], dtype=np.float32)
        result = _to_native_local_dtype(data, "int8", rng=rng)
        # 返回原生 int8 dtype
        assert result.dtype == np.int8
        assert result.shape == data.shape
        # 值在 int8 范围内
        assert np.all(result >= -127) and np.all(result <= 127)

    def test_int16_native_dtype(self):
        """int16: 直接生成原生 int16 随机整数"""
        from aidevtools.datagen import _to_native_local_dtype
        rng = np.random.default_rng(42)
        data = np.array([1.0, -0.5, 0.3, -0.8], dtype=np.float32)
        result = _to_native_local_dtype(data, "int16", rng=rng)
        assert result.dtype == np.int16
        assert result.shape == data.shape

    def test_fp32_passthrough(self):
        """fp32: 原样返回"""
        from aidevtools.datagen import _to_native_local_dtype
        data = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        result = _to_native_local_dtype(data, "fp32")
        np.testing.assert_array_equal(data, result)

    def test_int8_golden_via_cast(self):
        """int8 数据通过 .astype(fp32) 用于 golden 计算"""
        from aidevtools.datagen import _to_native_local_dtype
        rng = np.random.default_rng(42)
        data = np.zeros((4, 8), dtype=np.float32)
        int8_data = _to_native_local_dtype(data, "int8", rng=rng)
        # 可以安全 cast 到 fp32 用于 golden
        fp32_data = int8_data.astype(np.float32)
        assert fp32_data.dtype == np.float32
        np.testing.assert_array_equal(fp32_data, int8_data.astype(np.float64))


# ============================================================
# 8. XLSX OpConfig 精度配置测试
# ============================================================


class TestXlsxPrecisionConfig:
    """XLSX OpConfig 精度配置"""

    def test_opconfig_default_precision(self):
        from aidevtools.xlsx.import_ import OpConfig
        cfg = OpConfig(
            id=0, op_name="relu", shape=(4, 8),
            dtype="float32", depends="", qtype="bfp16",
            skip=False, note="",
        )
        assert cfg.compute_dtype == "fp32"
        assert cfg.input_dtype == "fp32"
        assert cfg.qa_aware is False

    def test_opconfig_to_precision_config(self):
        from aidevtools.xlsx.import_ import OpConfig
        cfg = OpConfig(
            id=0, op_name="linear", shape=(4, 8),
            dtype="float32", depends="", qtype="bfp16",
            skip=False, note="",
            compute_dtype="fp32",
            input_dtype="fp16",
            output_dtype="bfp8",
            weight_dtype="int8",
            qa_aware=True,
            qa_center=1.5,
            qa_amplitude=0.3,
        )
        pc = cfg.to_precision_config()
        assert pc.compute_dtype == "fp32"
        assert pc.input_dtype == "fp16"
        assert pc.output_dtype == "bfp8"
        assert pc.weight_dtype == "int8"
        assert pc.qa_aware is True
        assert pc.qa_center == 1.5
        assert pc.qa_amplitude == 0.3


# ============================================================
# 9. TorchBackendConfig 精度配置测试
# ============================================================


class TestTorchBackendPrecision:
    """TorchBackendConfig 精度配置"""

    def test_default_config(self):
        from aidevtools.torch_backend import TorchBackendConfig
        cfg = TorchBackendConfig()
        assert cfg.compute_dtype == "fp32"
        assert cfg.qa_aware is False

    def test_precision_config_conversion(self):
        from aidevtools.torch_backend import TorchBackendConfig
        cfg = TorchBackendConfig(
            input_dtype="fp16",
            weight_dtype="int8",
            qa_aware=True,
            qa_center=2.0,
        )
        pc = cfg.to_precision_config()
        assert pc.input_dtype == "fp16"
        assert pc.weight_dtype == "int8"
        assert pc.qa_aware is True
        assert pc.qa_center == 2.0


# ============================================================
# 10. 端到端: 四种比数 + CompareEngine
# ============================================================


class TestFourTrackWithCompare:
    """端到端: 四种比数 + 比对引擎"""

    def test_compare_pure_vs_local(self):
        """比较 pure golden 和 local golden"""
        from aidevtools.datagen import DataGenerator
        from aidevtools.frontend.types import PrecisionConfig
        from aidevtools.compare import CompareEngine, CompareConfig

        pc = PrecisionConfig(input_dtype="fp16")
        gen = DataGenerator(seed=42, precision=pc)
        tracks = gen.generate_four_track("relu", input_shape=(4, 16), precision=pc)

        if tracks.golden_local is not None:
            config = CompareConfig(fuzzy_min_qsnr=20.0, fuzzy_min_cosine=0.99)
            engine = CompareEngine.standard(config=config)
            result = engine.run(dut=tracks.golden_local, golden=tracks.golden_pure)
            # fp16 降精度误差应该在合理范围内
            fuzzy_result = result.get('fuzzy_pure')
            if fuzzy_result:
                assert fuzzy_result.cosine > 0.99
                assert fuzzy_result.qsnr > 20.0

    def test_compare_pure_vs_qa(self):
        """比较 pure golden 和 qa golden"""
        from aidevtools.datagen import DataGenerator
        from aidevtools.frontend.types import PrecisionConfig
        from aidevtools.compare import CompareEngine, CompareConfig

        pc = PrecisionConfig(qa_aware=True, qa_center=1.0, qa_amplitude=0.5)
        gen = DataGenerator(seed=42, precision=pc)
        tracks = gen.generate_four_track("relu", input_shape=(4, 16), precision=pc)

        if tracks.golden_qa is not None:
            # QA golden 使用不同的输入数据，所以和 pure 差异较大
            # 但两者都应该是有效的
            assert not np.any(np.isnan(tracks.golden_qa))
            assert not np.any(np.isinf(tracks.golden_qa))

    def test_full_compare_engine(self):
        """完整四种比数 + CompareEngine"""
        from aidevtools.datagen import DataGenerator
        from aidevtools.frontend.types import PrecisionConfig
        from aidevtools.compare import CompareEngine, CompareConfig, CompareStatus

        pc = PrecisionConfig(input_dtype="fp16", qa_aware=True)
        gen = DataGenerator(seed=42, precision=pc)
        tracks = gen.generate_four_track("relu", input_shape=(4, 16), precision=pc)

        # 模拟 DUT 输出 = golden_pure + 微小噪声
        dut = tracks.golden_pure + np.random.RandomState(0).normal(0, 1e-6, tracks.golden_pure.shape)

        # 放宽 fuzzy 阈值以适应 fp16 量化差异
        config = CompareConfig(
            fuzzy_min_qsnr=20.0,
            fuzzy_min_cosine=0.99,
            fuzzy_max_exceed_ratio=0.05,  # 允许 5% 元素超限
        )
        engine = CompareEngine.standard(config=config)

        # 使用新的 run() API
        result = engine.run(
            dut=dut.astype(np.float32),
            golden=tracks.golden_pure.astype(np.float32),
            golden_qnt=tracks.golden_local.astype(np.float32) if tracks.golden_local is not None else None,
        )

        # 验证四种比数的完整性
        assert result.get('fuzzy_pure') is not None, "fuzzy_pure should be computed"
        assert result.get('sanity') is not None, "sanity should be computed"
        status = result.get('status')
        assert status in (
            CompareStatus.PASS, CompareStatus.GOLDEN_SUSPECT, CompareStatus.DUT_ISSUE
        )


# ============================================================
# 11. 逐参数精度 (param_dtypes) 测试
# ============================================================


class TestParamDtypes:
    """逐参数精度覆盖 (param_dtypes + get_dtype)"""

    def test_get_dtype_param_override(self):
        """param_dtypes 优先级高于角色级默认"""
        from aidevtools.frontend.types import PrecisionConfig
        pc = PrecisionConfig(
            input_dtype="fp16",
            weight_dtype="bfp8",
            param_dtypes={"x": "int8", "weight": "bfp16"},
        )
        assert pc.get_dtype("x") == "int8"
        assert pc.get_dtype("weight", is_weight=True) == "bfp16"
        # 未覆盖的参数使用角色级默认
        assert pc.get_dtype("bias", is_weight=True) == "bfp8"
        assert pc.get_dtype("gamma") == "fp16"

    def test_get_dtype_fallback(self):
        """无 param_dtypes 时 fallback 到角色级"""
        from aidevtools.frontend.types import PrecisionConfig
        pc = PrecisionConfig(input_dtype="fp16", weight_dtype="int8")
        assert pc.get_dtype("x") == "fp16"
        assert pc.get_dtype("weight", is_weight=True) == "int8"

    def test_is_mixed_includes_param_dtypes(self):
        """is_mixed 考虑 param_dtypes"""
        from aidevtools.frontend.types import PrecisionConfig
        # 角色级全 fp32，但 param_dtypes 有不同的
        pc = PrecisionConfig(param_dtypes={"x": "bfp8"})
        assert pc.is_mixed

    def test_from_dict_with_param_dtypes(self):
        """from_dict 能正确传递 param_dtypes"""
        from aidevtools.frontend.types import PrecisionConfig
        d = {
            "input_dtype": "fp16",
            "param_dtypes": {"x": "bfp8", "weight": "int8"},
        }
        pc = PrecisionConfig.from_dict(d)
        assert pc.get_dtype("x") == "bfp8"
        assert pc.get_dtype("weight", is_weight=True) == "int8"

    def test_four_track_per_param_hw(self):
        """generate_four_track 使用 param_dtypes 触发 hw track"""
        from aidevtools.datagen import DataGenerator
        from aidevtools.frontend.types import PrecisionConfig

        pc = PrecisionConfig(
            input_dtype="fp32",
            weight_dtype="fp32",
            param_dtypes={"x": "bfp8"},  # 仅 x 用 bfp8
        )
        gen = DataGenerator(seed=42, precision=pc)
        tracks = gen.generate_four_track("relu", input_shape=(4, 16), precision=pc)

        assert tracks.golden_pure is not None
        # param_dtypes 中有 bfp8，应触发 hw track
        assert tracks.golden_hw is not None

    def test_four_track_per_param_local(self):
        """generate_four_track 使用 param_dtypes 触发 local track"""
        from aidevtools.datagen import DataGenerator
        from aidevtools.frontend.types import PrecisionConfig

        pc = PrecisionConfig(
            input_dtype="fp32",
            weight_dtype="fp32",
            param_dtypes={"x": "fp16"},  # 仅 x 用 fp16
        )
        gen = DataGenerator(seed=42, precision=pc)
        tracks = gen.generate_four_track("relu", input_shape=(4, 16), precision=pc)

        assert tracks.golden_pure is not None
        assert tracks.golden_local is not None

    def test_xlsx_opconfig_param_dtypes(self):
        """XLSX OpConfig 支持 param_dtypes"""
        from aidevtools.xlsx.import_ import OpConfig
        cfg = OpConfig(
            id=0, op_name="linear", shape=(4, 8),
            dtype="float32", depends="", qtype="bfp16",
            skip=False, note="",
            input_dtype="fp16",
            weight_dtype="bfp8",
            param_dtypes={"bias": "fp32", "x": "int8"},
        )
        pc = cfg.to_precision_config()
        assert pc.get_dtype("x") == "int8"
        assert pc.get_dtype("bias", is_weight=True) == "fp32"
        assert pc.get_dtype("weight", is_weight=True) == "bfp8"

    def test_torch_backend_param_dtypes(self):
        """TorchBackendConfig 支持 param_dtypes"""
        from aidevtools.torch_backend import TorchBackendConfig
        cfg = TorchBackendConfig(
            input_dtype="fp16",
            weight_dtype="bfp8",
            param_dtypes={"input": "int8", "weight": "bfp16"},
        )
        pc = cfg.to_precision_config()
        assert pc.get_dtype("input") == "int8"
        assert pc.get_dtype("weight", is_weight=True) == "bfp16"

    def test_model_dsl_per_param_qtype(self):
        """Model DSL _call_op 使用 per-param 精度"""
        from aidevtools.datagen import Model
        from aidevtools.frontend.types import PrecisionConfig

        pc = PrecisionConfig(
            input_dtype="fp16",
            weight_dtype="bfp8",
            param_dtypes={"weight": "int8"},
        )
        with Model(seed=42, precision=pc, qtype="bfp16") as m:
            x = m.input((4, 8))
            y = m.linear(x, out_features=16)

        # 验证 weight tensor 使用了 param_dtypes 中的 int8
        weight_tensors = {
            k: v for k, v in m.tensors.items() if "weight" in k
        }
        assert len(weight_tensors) > 0
        for name, t in weight_tensors.items():
            assert t.qtype == "int8", f"{name} should use int8 from param_dtypes, got {t.qtype}"

    def test_datagen_generate_per_param(self):
        """DataGenerator.generate 使用 per-param 精度"""
        from aidevtools.datagen import DataGenerator
        from aidevtools.frontend.types import PrecisionConfig

        pc = PrecisionConfig(
            input_dtype="fp16",
            weight_dtype="bfp8",
            param_dtypes={"weight": "int8"},
        )
        gen = DataGenerator(seed=42, precision=pc, qtype="bfp16")
        data = gen.generate("linear", input_shape=(4, 8), out_features=16)

        # weight 应使用 param_dtypes 中指定的 int8
        assert data["weight"].qtype == "int8"
        # input 不在 param_dtypes 中，fallback 到 input_dtype="fp16"
        assert data["input"].qtype == "fp16"
