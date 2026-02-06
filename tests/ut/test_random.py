"""Tests for aidevtools.core.random.RandomGenerator"""

import numpy as np
import pytest

from aidevtools.core.random import Method, RandomGenerator, parse_shape


class TestRandomGeneratorBasic:
    """基础功能测试"""

    def test_default_seed_is_none(self):
        rng = RandomGenerator()
        assert rng.seed is None

    def test_seed_deterministic(self):
        a = RandomGenerator(seed=42).generate((3, 4), method="normal")
        b = RandomGenerator(seed=42).generate((3, 4), method="normal")
        np.testing.assert_array_equal(a, b)

    def test_different_seeds_differ(self):
        a = RandomGenerator(seed=1).generate((100,), method="normal")
        b = RandomGenerator(seed=2).generate((100,), method="normal")
        assert not np.array_equal(a, b)

    def test_reset_restores_state(self):
        rng = RandomGenerator(seed=7)
        first = rng.generate((5,), method="normal")
        rng.reset()
        second = rng.generate((5,), method="normal")
        np.testing.assert_array_equal(first, second)

    def test_reset_with_new_seed(self):
        rng = RandomGenerator(seed=7)
        rng.reset(seed=99)
        assert rng.seed == 99


class TestMethods:
    """各方法输出正确性"""

    @pytest.fixture
    def rng(self):
        return RandomGenerator(seed=42)

    def test_normal_shape_dtype(self, rng):
        arr = rng.normal((2, 3))
        assert arr.shape == (2, 3)
        assert arr.dtype == np.float32

    def test_normal_custom_params(self, rng):
        arr = rng.normal((10000,), mean=5.0, std=0.01)
        assert abs(arr.mean() - 5.0) < 0.1

    def test_uniform_range(self, rng):
        arr = rng.uniform((10000,), low=-2.0, high=2.0)
        assert arr.min() >= -2.0
        assert arr.max() <= 2.0

    def test_zeros(self, rng):
        arr = rng.zeros((4, 5))
        np.testing.assert_array_equal(arr, np.zeros((4, 5), dtype=np.float32))

    def test_ones(self, rng):
        arr = rng.ones((3,))
        np.testing.assert_array_equal(arr, np.ones((3,), dtype=np.float32))

    def test_xavier_shape(self, rng):
        arr = rng.xavier((64, 128))
        assert arr.shape == (64, 128)
        limit = np.sqrt(6.0 / (128 + 64))
        assert arr.min() >= -limit - 1e-7
        assert arr.max() <= limit + 1e-7

    def test_kaiming_shape(self, rng):
        arr = rng.kaiming((64, 128))
        assert arr.shape == (64, 128)
        expected_std = np.sqrt(2.0 / 128)
        assert abs(arr.std() - expected_std) < 0.1


class TestDtype:
    """dtype 转换测试"""

    def test_default_fp32(self):
        arr = RandomGenerator(seed=0).generate((2,), method="normal")
        assert arr.dtype == np.float32

    def test_fp16_string(self):
        arr = RandomGenerator(seed=0).generate((2,), method="normal", dtype="fp16")
        assert arr.dtype == np.float16

    def test_float16_string(self):
        arr = RandomGenerator(seed=0).generate((2,), method="normal", dtype="float16")
        assert arr.dtype == np.float16

    def test_float64_numpy(self):
        arr = RandomGenerator(seed=0).generate((2,), method="normal", dtype=np.float64)
        assert arr.dtype == np.float64

    def test_none_defaults_to_fp32(self):
        arr = RandomGenerator(seed=0).generate((2,), method="normal", dtype=None)
        assert arr.dtype == np.float32


class TestMethodEnum:
    """Method 枚举测试"""

    def test_string_dispatch(self):
        rng = RandomGenerator(seed=0)
        for m in Method:
            arr = rng.generate((2, 2), method=m.value)
            assert arr.shape == (2, 2)

    def test_enum_dispatch(self):
        rng = RandomGenerator(seed=0)
        for m in Method:
            arr = rng.generate((2, 2), method=m)
            assert arr.shape == (2, 2)

    def test_invalid_method(self):
        rng = RandomGenerator(seed=0)
        with pytest.raises(ValueError, match="未知的随机方法"):
            rng.generate((2,), method="banana")


class TestGenerateEqualsShortcuts:
    """generate() 和便捷方法的等价性"""

    def test_normal_equiv(self):
        a = RandomGenerator(seed=1).generate((3,), method="normal")
        b = RandomGenerator(seed=1).normal((3,))
        np.testing.assert_array_equal(a, b)

    def test_uniform_equiv(self):
        a = RandomGenerator(seed=1).generate((3,), method="uniform", low=0, high=1)
        b = RandomGenerator(seed=1).uniform((3,), low=0, high=1)
        np.testing.assert_array_equal(a, b)

    def test_xavier_equiv(self):
        a = RandomGenerator(seed=1).generate((4, 5), method="xavier")
        b = RandomGenerator(seed=1).xavier((4, 5))
        np.testing.assert_array_equal(a, b)

    def test_kaiming_equiv(self):
        a = RandomGenerator(seed=1).generate((4, 5), method="kaiming")
        b = RandomGenerator(seed=1).kaiming((4, 5))
        np.testing.assert_array_equal(a, b)


# ============================================================
# parse_shape 测试
# ============================================================


class TestParseShape:
    """parse_shape() 独立函数测试"""

    def test_negative_index(self):
        ctx = {"input_shape": (2, 64, 128)}
        assert parse_shape("-1", ctx) == (128,)

    def test_multiple_indices(self):
        ctx = {"input_shape": (2, 64, 128)}
        assert parse_shape("-2,-1", ctx) == (64, 128)

    def test_named_param(self):
        ctx = {"input_shape": (2, 64), "out_features": 256}
        assert parse_shape("out_features", ctx) == (256,)

    def test_mixed(self):
        ctx = {"input_shape": (2, 64), "out_features": 256}
        assert parse_shape("out_features,-1", ctx) == (256, 64)

    def test_tuple_expansion(self):
        ctx = {"input_shape": (2, 64), "sizes": (3, 4)}
        assert parse_shape("sizes", ctx) == (3, 4)

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="无法解析"):
            parse_shape("nonexistent", {"input_shape": (1,)})

    def test_empty_parts_skipped(self):
        ctx = {"input_shape": (2, 64, 128)}
        assert parse_shape("-1,", ctx) == (128,)


# ============================================================
# generate_from_strategy 测试
# ============================================================


class TestGenerateFromStrategy:
    """策略字符串解析测试"""

    @pytest.fixture
    def rng(self):
        return RandomGenerator(seed=42)

    def test_input_strategy(self, rng):
        ctx = {"input_shape": (2, 64)}
        data, shape = rng.generate_from_strategy("input", ctx)
        assert shape == (2, 64)
        assert data.shape == (2, 64)

    def test_random_strategy(self, rng):
        ctx = {"input_shape": (3, 32)}
        data, shape = rng.generate_from_strategy("random", ctx)
        assert shape == (3, 32)

    def test_xavier_simple(self, rng):
        ctx = {"input_shape": (2, 64), "out_features": 128}
        data, shape = rng.generate_from_strategy("xavier", ctx)
        assert shape == (128, 64)

    def test_xavier_with_shape(self, rng):
        ctx = {"input_shape": (2, 64), "out_features": 128}
        data, shape = rng.generate_from_strategy("xavier:out_features,-1", ctx)
        assert shape == (128, 64)

    def test_kaiming_simple(self, rng):
        ctx = {"input_shape": (2, 64), "out_features": 128}
        data, shape = rng.generate_from_strategy("kaiming", ctx)
        assert shape == (128, 64)

    def test_uniform_simple(self, rng):
        ctx = {"input_shape": (2, 64), "out_features": 128}
        data, shape = rng.generate_from_strategy("uniform", ctx)
        assert shape == (128,)
        assert data.min() >= -0.1
        assert data.max() <= 0.1

    def test_zeros_strategy(self, rng):
        ctx = {"input_shape": (2, 64)}
        data, shape = rng.generate_from_strategy("zeros:-1", ctx)
        assert shape == (64,)
        np.testing.assert_array_equal(data, np.zeros((64,), dtype=np.float32))

    def test_ones_strategy(self, rng):
        ctx = {"input_shape": (2, 64)}
        data, shape = rng.generate_from_strategy("ones:-1", ctx)
        assert shape == (64,)
        np.testing.assert_array_equal(data, np.ones((64,), dtype=np.float32))

    def test_same_strategy(self, rng):
        ctx = {"input_shape": (2, 64), "weight_shape": (128, 64)}
        data, shape = rng.generate_from_strategy("same:weight", ctx)
        assert shape == (128, 64)

    def test_same_fallback_to_input(self, rng):
        ctx = {"input_shape": (2, 64)}
        data, shape = rng.generate_from_strategy("same:missing", ctx)
        assert shape == (2, 64)

    def test_normal_with_params(self, rng):
        ctx = {"input_shape": (2, 64)}
        data, shape = rng.generate_from_strategy("normal:5.0,0.01,-1", ctx)
        assert shape == (64,)
        assert abs(data.mean() - 5.0) < 1.0  # 小样本容许宽松

    def test_uniform_with_params(self, rng):
        ctx = {"input_shape": (2, 64)}
        data, shape = rng.generate_from_strategy("uniform:-2.0,2.0,-1", ctx)
        assert shape == (64,)
        assert data.min() >= -2.0
        assert data.max() <= 2.0

    def test_unknown_strategy_raises(self, rng):
        with pytest.raises(ValueError, match="未知生成策略"):
            rng.generate_from_strategy("foobar", {"input_shape": (1,)})

    def test_deterministic(self):
        """相同 seed 的 strategy 结果一致"""
        ctx = {"input_shape": (4, 64), "out_features": 128}
        a, _ = RandomGenerator(seed=7).generate_from_strategy("xavier", ctx)
        b, _ = RandomGenerator(seed=7).generate_from_strategy("xavier", ctx)
        np.testing.assert_array_equal(a, b)


# ============================================================
# qtype (量化模拟) 测试
# ============================================================


class TestQuantizeSimulation:
    """qtype 参数 (quantize→dequantize) 测试"""

    def test_qtype_none_no_change(self):
        """qtype=None 不做任何量化"""
        rng = RandomGenerator(seed=42)
        a = rng.generate((64,), method="normal", qtype=None)
        rng.reset()
        b = rng.generate((64,), method="normal")
        np.testing.assert_array_equal(a, b)

    def test_qtype_fp32_no_change(self):
        """qtype='fp32' 不做量化"""
        rng = RandomGenerator(seed=42)
        a = rng.generate((64,), method="normal", qtype="fp32")
        rng.reset()
        b = rng.generate((64,), method="normal")
        np.testing.assert_array_equal(a, b)

    def test_qtype_bfp16_introduces_loss(self):
        """qtype='bfp16' 经过量化→反量化后数据有精度损失"""
        rng = RandomGenerator(seed=42)
        exact = rng.generate((256,), method="normal")
        rng.reset()
        lossy = rng.generate((256,), method="normal", qtype="bfp16")
        # 形状一致
        assert exact.shape == lossy.shape
        # 不完全相等 (有量化损失)
        assert not np.array_equal(exact, lossy)
        # 但应该接近
        np.testing.assert_allclose(exact, lossy, rtol=0.05, atol=0.05)

    def test_qtype_bfp8_larger_loss(self):
        """bfp8 的精度损失大于 bfp16"""
        rng = RandomGenerator(seed=42)
        exact = rng.generate((256,), method="normal")

        rng.reset()
        lossy_16 = rng.generate((256,), method="normal", qtype="bfp16")

        rng.reset()
        lossy_8 = rng.generate((256,), method="normal", qtype="bfp8")

        err_16 = np.abs(exact - lossy_16).mean()
        err_8 = np.abs(exact - lossy_8).mean()
        assert err_8 > err_16

    def test_strategy_with_qtype(self):
        """generate_from_strategy 也支持 qtype"""
        rng = RandomGenerator(seed=42)
        ctx = {"input_shape": (2, 64)}
        data_exact, _ = rng.generate_from_strategy("input", ctx)

        rng.reset()
        data_lossy, _ = rng.generate_from_strategy("input", ctx, qtype="bfp16")

        assert data_exact.shape == data_lossy.shape
        assert not np.array_equal(data_exact, data_lossy)
