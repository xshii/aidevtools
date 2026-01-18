"""量化模块单元测试"""
import pytest
import numpy as np


class TestQuantizeRegistry:
    """量化注册表测试"""

    def test_list_quantize(self):
        """列出所有量化类型"""
        from aidevtools.formats.quantize import list_quantize

        qtypes = list_quantize()
        assert "float16" in qtypes
        assert "gfloat16" in qtypes
        assert "gfloat8" in qtypes

    def test_register_quantize(self):
        """注册自定义量化类型"""
        from aidevtools.formats.quantize import register_quantize, list_quantize

        @register_quantize("test_qtype")
        def to_test(data, **kwargs):
            return data.astype(np.int8), {"test": True}

        assert "test_qtype" in list_quantize()

    def test_get_unknown_quantize(self):
        """获取未知量化类型"""
        from aidevtools.formats.quantize import get_quantize

        with pytest.raises(ValueError, match="未知量化类型"):
            get_quantize("unknown_qtype")


class TestBuiltinQuantize:
    """内置量化类型测试"""

    def test_float16(self):
        """float16 量化"""
        from aidevtools.formats.quantize import quantize

        data = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        q_data, meta = quantize(data, "float16")

        assert q_data.dtype == np.float16
        assert np.allclose(q_data, data, atol=1e-3)
        assert meta == {}

    def test_int8_symmetric_not_implemented(self):
        """int8 对称量化未实现"""
        from aidevtools.formats.quantize import quantize

        data = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        with pytest.raises(NotImplementedError):
            quantize(data, "int8_symmetric")


class TestGfloat:
    """Gfloat 自定义格式测试"""

    def test_gfloat16(self):
        """gfloat16 量化"""
        from aidevtools.formats.quantize import quantize

        data = np.array([1.0, 2.0, -3.0], dtype=np.float32)
        q_data, meta = quantize(data, "gfloat16")

        assert q_data.dtype == np.uint16
        assert meta["format"] == "gfloat16_as_uint16"
        # gfloat16 取 fp32 高 16 位
        expected = (data.view(np.uint32) >> 16).astype(np.uint16)
        assert np.array_equal(q_data, expected)

    def test_gfloat8(self):
        """gfloat8 量化"""
        from aidevtools.formats.quantize import quantize

        data = np.array([1.0, 2.0, -3.0], dtype=np.float32)
        q_data, meta = quantize(data, "gfloat8")

        assert q_data.dtype == np.uint8
        assert meta["format"] == "gfloat8_as_uint8"
        # gfloat8 取 fp32 高 8 位
        expected = (data.view(np.uint32) >> 24).astype(np.uint8)
        assert np.array_equal(q_data, expected)

    def test_gfloat16_roundtrip(self):
        """gfloat16 往返精度"""
        from aidevtools.formats.quantize import quantize

        # 选择能精确表示的值
        data = np.array([0.0, 1.0, -1.0, 2.0], dtype=np.float32)
        q_data, _ = quantize(data, "gfloat16")

        # 还原：低 16 位补 0
        restored = (q_data.astype(np.uint32) << 16).view(np.float32)
        # gfloat16 会损失精度，但符号和数量级应保持
        assert np.allclose(restored, data, rtol=1e-2)
