"""Golden API 单元测试"""
import pytest
import numpy as np


def _is_gfloat_cpp_available():
    """检查 GFloat C++ 是否可用"""
    try:
        from aidevtools.formats.custom.gfloat.wrapper import is_cpp_available
        return is_cpp_available()
    except ImportError:
        return False


class TestGfloatGolden:
    """GFloat Golden API 测试"""

    @pytest.mark.skipif(not _is_gfloat_cpp_available(), reason="GFloat C++ 扩展未编译")
    def test_fp32_to_gfloat16(self):
        """fp32 -> gfloat16 转换"""
        from aidevtools.formats.custom.gfloat.wrapper import fp32_to_gfloat16

        data = np.array([1.0, 2.0, -3.0, 0.0], dtype=np.float32)
        result = fp32_to_gfloat16(data)

        assert result.dtype == np.uint16
        assert result.shape == data.shape

        # 验证转换正确性 (取高16位)
        expected = (data.view(np.uint32) >> 16).astype(np.uint16)
        assert np.array_equal(result, expected)

    @pytest.mark.skipif(not _is_gfloat_cpp_available(), reason="GFloat C++ 扩展未编译")
    def test_gfloat16_to_fp32(self):
        """gfloat16 -> fp32 转换"""
        from aidevtools.formats.custom.gfloat.wrapper import fp32_to_gfloat16, gfloat16_to_fp32

        original = np.array([1.0, 2.0, -3.0, 0.0], dtype=np.float32)
        gf16 = fp32_to_gfloat16(original)
        restored = gfloat16_to_fp32(gf16)

        assert restored.dtype == np.float32
        # 往返后应该接近原值 (有精度损失)
        assert np.allclose(restored, original, rtol=1e-2)

    @pytest.mark.skipif(not _is_gfloat_cpp_available(), reason="GFloat C++ 扩展未编译")
    def test_fp32_to_gfloat8(self):
        """fp32 -> gfloat8 转换"""
        from aidevtools.formats.custom.gfloat.wrapper import fp32_to_gfloat8

        data = np.array([1.0, 2.0, -3.0, 0.0], dtype=np.float32)
        result = fp32_to_gfloat8(data)

        assert result.dtype == np.uint8
        assert result.shape == data.shape

        # 验证转换正确性 (取高8位)
        expected = (data.view(np.uint32) >> 24).astype(np.uint8)
        assert np.array_equal(result, expected)

    @pytest.mark.skipif(not _is_gfloat_cpp_available(), reason="GFloat C++ 扩展未编译")
    def test_gfloat8_to_fp32(self):
        """gfloat8 -> fp32 转换"""
        from aidevtools.formats.custom.gfloat.wrapper import fp32_to_gfloat8, gfloat8_to_fp32

        original = np.array([1.0, 2.0, -3.0, 0.0], dtype=np.float32)
        gf8 = fp32_to_gfloat8(original)
        restored = gfloat8_to_fp32(gf8)

        assert restored.dtype == np.float32
        # gfloat8 精度很低，只检查符号正确
        assert np.sign(restored[0]) == np.sign(original[0])
        assert np.sign(restored[2]) == np.sign(original[2])

    @pytest.mark.skipif(not _is_gfloat_cpp_available(), reason="GFloat C++ 扩展未编译")
    def test_multidim_array(self):
        """多维数组测试"""
        from aidevtools.formats.custom.gfloat.wrapper import fp32_to_gfloat16, gfloat16_to_fp32

        data = np.random.randn(2, 3, 4).astype(np.float32)
        gf16 = fp32_to_gfloat16(data)
        restored = gfloat16_to_fp32(gf16)

        assert gf16.shape == data.shape
        assert restored.shape == data.shape
        assert np.allclose(restored, data, rtol=1e-2)

    def test_is_cpp_available(self):
        """检查 C++ 可用性"""
        from aidevtools.formats.custom.gfloat.wrapper import is_cpp_available

        # 不管是否可用，函数应该返回 bool
        result = is_cpp_available()
        assert isinstance(result, bool)

    def test_check_cpp_error(self):
        """C++ 不可用时报错"""
        from aidevtools.formats.custom.gfloat import wrapper as gfloat_wrapper

        # 模拟 C++ 不可用
        loader = gfloat_wrapper._loader
        original_module = loader._module
        original_error = loader._import_error

        try:
            loader._module = None
            loader._import_error = ImportError("test error")
            loader._import_detail = "测试错误"

            with pytest.raises(ImportError, match="加载失败"):
                loader.check()
        finally:
            # 恢复
            loader._module = original_module
            loader._import_error = original_error


class TestGfloatGoldenRegister:
    """Golden 注册测试"""

    @pytest.mark.skipif(not _is_gfloat_cpp_available(), reason="GFloat C++ 扩展未编译")
    def test_register_golden(self):
        """注册 golden 实现"""
        from aidevtools.formats.custom.gfloat import register_gfloat_golden
        from aidevtools.formats.quantize import list_quantize

        register_gfloat_golden()

        qtypes = list_quantize()
        assert "gfloat16_golden" in qtypes
        assert "gfloat8_golden" in qtypes

    @pytest.mark.skipif(not _is_gfloat_cpp_available(), reason="GFloat C++ 扩展未编译")
    def test_use_golden_quantize(self):
        """使用 golden 量化"""
        from aidevtools.formats.custom.gfloat import register_gfloat_golden
        from aidevtools.formats.quantize import quantize

        register_gfloat_golden()

        data = np.array([1.0, 2.0, 3.0], dtype=np.float32)

        result16, meta16 = quantize(data, "gfloat16_golden")
        assert result16.dtype == np.uint16
        assert "cpp" in meta16

        result8, meta8 = quantize(data, "gfloat8_golden")
        assert result8.dtype == np.uint8
        assert "cpp" in meta8
