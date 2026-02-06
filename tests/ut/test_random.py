"""Tests for aidevtools.core.random.RandomGenerator"""

import numpy as np
import pytest

from aidevtools.core.random import Method, RandomGenerator


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
        # Xavier 初始化范围 ∝ sqrt(6 / (fan_in + fan_out))
        limit = np.sqrt(6.0 / (128 + 64))
        assert arr.min() >= -limit - 1e-7
        assert arr.max() <= limit + 1e-7

    def test_kaiming_shape(self, rng):
        arr = rng.kaiming((64, 128))
        assert arr.shape == (64, 128)
        # Kaiming std ∝ sqrt(2 / fan_in)
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
