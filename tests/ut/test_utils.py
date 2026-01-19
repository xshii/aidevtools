"""测试 core.utils 工具函数"""
import pytest
import numpy as np
from aidevtools.core.utils import (
    parse_shape,
    parse_dtype,
    parse_list,
    format_shape,
    safe_getattr,
)


class TestParseShape:
    """测试 parse_shape 函数"""

    def test_normal_shape(self):
        """正常 shape 解析"""
        assert parse_shape("1,64,32,32") == (1, 64, 32, 32)

    def test_shape_with_spaces(self):
        """带空格的 shape"""
        assert parse_shape("1, 64, 32") == (1, 64, 32)
        assert parse_shape(" 1 , 64 , 32 ") == (1, 64, 32)

    def test_empty_shape(self):
        """空字符串返回 None"""
        assert parse_shape("") is None
        assert parse_shape("   ") is None

    def test_single_dimension(self):
        """单维度"""
        assert parse_shape("128") == (128,)


class TestParseDtype:
    """测试 parse_dtype 函数"""

    def test_common_dtypes(self):
        """常见 dtype 解析"""
        assert parse_dtype("float32") == np.float32
        assert parse_dtype("float16") == np.float16
        assert parse_dtype("int32") == np.int32
        assert parse_dtype("uint8") == np.uint8

    def test_invalid_dtype(self):
        """无效 dtype 应抛出异常"""
        with pytest.raises(AttributeError):
            parse_dtype("invalid_dtype")


class TestParseList:
    """测试 parse_list 函数"""

    def test_normal_list(self):
        """正常列表解析"""
        assert parse_list("a,b,c") == ["a", "b", "c"]

    def test_list_with_spaces(self):
        """带空格的列表"""
        assert parse_list("a, b, c") == ["a", "b", "c"]
        assert parse_list(" a , b , c ") == ["a", "b", "c"]

    def test_empty_list(self):
        """空字符串返回空列表"""
        assert parse_list("") == []
        assert parse_list("   ") == []

    def test_single_item(self):
        """单元素"""
        assert parse_list("item") == ["item"]

    def test_custom_separator(self):
        """自定义分隔符"""
        assert parse_list("a;b;c", separator=";") == ["a", "b", "c"]


class TestFormatShape:
    """测试 format_shape 函数"""

    def test_format_shape(self):
        """格式化 shape"""
        assert format_shape((1, 64, 32, 32)) == "1,64,32,32"
        assert format_shape((128,)) == "128"

    def test_empty_shape(self):
        """空 shape"""
        assert format_shape(()) == ""


class TestSafeGetattr:
    """测试 safe_getattr 函数"""

    def test_existing_attr(self):
        """存在的属性"""
        assert safe_getattr(np, "float32") == np.float32

    def test_missing_attr(self):
        """不存在的属性返回默认值"""
        assert safe_getattr(np, "nonexistent", "default") == "default"
        assert safe_getattr(np, "nonexistent") is None
