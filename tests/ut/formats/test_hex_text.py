"""Hex-text 格式单元测试"""
import numpy as np
import pytest

from aidevtools.formats._registry import list_formats
from aidevtools.formats.base import load, save
from aidevtools.formats._registry import get


class TestHexTextRegistered:
    """注册测试"""

    def test_registered(self):
        """hex_text 已注册"""
        assert "hex_text" in list_formats()


class TestHexTextSaveLoad:
    """保存 / 加载测试"""

    def test_save_load_roundtrip(self, tmp_workspace):
        """写入再读回 → 数据一致"""
        data = np.array([0x3F, 0x80, 0x00, 0x00, 0xFF, 0x01], dtype=np.uint8)
        path = str(tmp_workspace / "test.txt")
        save(path, data, fmt="hex_text")
        loaded = get("hex_text").load(path)
        assert np.array_equal(data, loaded)

    def test_load_lowercase_hex(self, tmp_workspace):
        """支持小写 hex"""
        path = str(tmp_workspace / "lower.txt")
        with open(path, "w") as f:
            f.write("3f\n80\n00\nff\n")
        loaded = get("hex_text").load(path)
        expected = np.array([0x3F, 0x80, 0x00, 0xFF], dtype=np.uint8)
        assert np.array_equal(loaded, expected)

    def test_load_with_dtype(self, tmp_workspace):
        """加载时转换 dtype"""
        data = np.array([1.0, 2.0], dtype=np.float32)
        path = str(tmp_workspace / "fp32.txt")
        save(path, data, fmt="hex_text")
        loaded = get("hex_text").load(path, dtype=np.float32)
        assert np.allclose(data, loaded)

    def test_load_with_shape(self, tmp_workspace):
        """加载时 reshape"""
        data = np.arange(6, dtype=np.uint8)
        path = str(tmp_workspace / "shaped.txt")
        save(path, data, fmt="hex_text")
        loaded = get("hex_text").load(path, shape=(2, 3))
        assert loaded.shape == (2, 3)
        assert np.array_equal(loaded.flatten(), data)

    def test_empty_lines_skipped(self, tmp_workspace):
        """空行被跳过"""
        path = str(tmp_workspace / "gaps.txt")
        with open(path, "w") as f:
            f.write("0A\n\n0B\n  \n0C\n")
        loaded = get("hex_text").load(path)
        assert np.array_equal(loaded, np.array([0x0A, 0x0B, 0x0C], dtype=np.uint8))


class TestHexTextWithQtype:
    """通过 base.load 全链路测试"""

    def test_load_with_qtype_bfp8(self, tmp_workspace):
        """hex_text + qtype=bfp8 全链路 → fp32 shape 正确"""
        from aidevtools.formats.quantize import quantize

        original = np.random.randn(64).astype(np.float32) * 0.5
        packed, meta = quantize(original, "bfpp8")
        # 将 packed 存为 hex-text
        path = str(tmp_workspace / "data.txt")
        save(path, packed, fmt="hex_text")
        # 通过 load 全链路: hex_text → bytes → dequantize → fp32
        loaded = load(path, fmt="hex_text", qtype="bfpp8", shape=(64,))
        assert loaded.dtype == np.float32
        assert loaded.shape == (64,)

    def test_load_with_qtype_gfloat8(self, tmp_workspace):
        """hex_text + qtype=gfloat8 全链路 → fp32"""
        from aidevtools.formats.quantize import quantize

        original = np.array([1.0, -0.5, 0.25, 3.0], dtype=np.float32)
        packed, meta = quantize(original, "gfloat8")
        path = str(tmp_workspace / "gfloat.txt")
        save(path, packed, fmt="hex_text")
        loaded = load(path, fmt="hex_text", qtype="gfloat8", shape=(4,))
        assert loaded.dtype == np.float32
        assert loaded.shape == (4,)
