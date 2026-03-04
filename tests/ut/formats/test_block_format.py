"""Block Format 注册框架单元测试"""
import numpy as np
import pytest

from aidevtools.formats.block_format import (
    BlockFormatSpec,
    register_block_format,
    get_block_format,
    is_block_format,
    list_block_formats,
    get_bit_layout,
    _registry,
)


# ── 测试用自定义格式 ──

def _dummy_quantize(data, **kwargs):
    """简单量化: 直接 round 到 int8"""
    block_size = kwargs.get("block_size", 4)
    flat = data.flatten().astype(np.float32)
    scale = np.max(np.abs(flat)) / 127.0 if np.max(np.abs(flat)) > 0 else 1.0
    quantized = np.round(flat / scale).astype(np.int8)
    meta = {
        "format": "dummy",
        "block_size": block_size,
        "mantissa_bits": 8,
        "num_blocks": len(flat) // block_size,
        "original_shape": data.shape,
        "scale": scale,
    }
    return quantized, meta


def _dummy_dequantize(data, meta):
    """简单反量化: int8 * scale"""
    scale = meta.get("scale", 1.0)
    original_shape = meta.get("original_shape", data.shape)
    result = data.astype(np.float32) * scale
    if original_shape is not None:
        result = result.reshape(original_shape)
    return result


@pytest.fixture(autouse=True)
def _cleanup_registry():
    """每个测试后清理自定义注册"""
    original_keys = set(_registry.keys())
    yield
    for key in list(_registry.keys()):
        if key not in original_keys:
            del _registry[key]


class TestRegistration:
    """注册功能测试"""

    def test_register_custom_format(self):
        """注册自定义格式"""
        spec = BlockFormatSpec(
            name="test_dummy8",
            block_size=4,
            mantissa_bits=8,
            quantize_fn=_dummy_quantize,
            dequantize_fn=_dummy_dequantize,
            description="Test dummy format",
        )
        register_block_format(spec)

        assert is_block_format("test_dummy8")
        assert get_block_format("test_dummy8") is spec
        assert "test_dummy8" in list_block_formats()

    def test_not_registered(self):
        """查询未注册格式"""
        assert not is_block_format("nonexistent_format")
        assert get_block_format("nonexistent_format") is None

    def test_builtin_bfpp_registered(self):
        """内置 BFPP 格式已注册"""
        assert is_block_format("bfpp16")
        assert is_block_format("bfpp8")
        assert is_block_format("bfpp4")

    def test_bfpp_spec_values(self):
        """BFPP spec 值正确"""
        spec = get_block_format("bfpp8")
        assert spec.block_size == 32
        assert spec.mantissa_bits == 4
        assert spec.storage_dtype == np.int8

        spec16 = get_block_format("bfpp16")
        assert spec16.block_size == 16
        assert spec16.mantissa_bits == 8

        spec4 = get_block_format("bfpp4")
        assert spec4.block_size == 64
        assert spec4.mantissa_bits == 2


class TestQuantizeDequantize:
    """通过 registry 自动注册的 quantize/dequantize 测试"""

    def test_quantize_via_registry(self):
        """注册后可通过 quantize() 调用"""
        register_block_format(BlockFormatSpec(
            name="test_q_fmt",
            block_size=4,
            mantissa_bits=8,
            quantize_fn=_dummy_quantize,
            dequantize_fn=_dummy_dequantize,
        ))

        from aidevtools.formats.quantize import quantize
        data = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        packed, meta = quantize(data, "test_q_fmt")
        assert packed.dtype == np.int8
        assert meta["format"] == "dummy"

    def test_dequantize_via_registry(self):
        """注册后可通过 dequantize() 调用"""
        register_block_format(BlockFormatSpec(
            name="test_dq_fmt",
            block_size=4,
            mantissa_bits=8,
            quantize_fn=_dummy_quantize,
            dequantize_fn=_dummy_dequantize,
        ))

        from aidevtools.formats.quantize import quantize, dequantize
        data = np.array([1.0, 2.0, -3.0, 0.5], dtype=np.float32)
        packed, meta = quantize(data, "test_dq_fmt")
        restored = dequantize(packed, "test_dq_fmt", meta)
        assert restored.dtype == np.float32
        assert restored.shape == data.shape
        assert np.allclose(restored, data, rtol=0.1, atol=0.1)

    def test_bfpp8_roundtrip(self):
        """BFPP8 通过 registry 的 roundtrip"""
        from aidevtools.formats.quantize import quantize, dequantize
        data = np.random.randn(64).astype(np.float32) * 0.5
        packed, meta = quantize(data, "bfpp8")
        restored = dequantize(packed, "bfpp8", meta)
        assert restored.shape == data.shape
        # bfpp8 只有 4 mantissa bits，用 simulate_quantize 验证一致性
        from aidevtools.formats.quantize import simulate_quantize
        expected = simulate_quantize(data, "bfpp8")
        np.testing.assert_array_equal(restored, expected)

    def test_simulate_quantize(self):
        """simulate_quantize 也能工作"""
        from aidevtools.formats.quantize import simulate_quantize
        data = np.random.randn(32).astype(np.float32)
        lossy = simulate_quantize(data, "bfpp8")
        assert lossy.shape == data.shape
        assert lossy.dtype == np.float32


class TestBitLayout:
    """get_bit_layout 测试"""

    def test_bfpp8_layout(self):
        """BFPP8 自动生成 BitLayout"""
        layout = get_bit_layout("bfpp8")
        assert layout.sign_bits == 1
        assert layout.exponent_bits == 0
        assert layout.mantissa_bits == 7  # int8 = 8 bits - 1 sign = 7
        assert layout.name == "bfpp8"
        assert layout.precision_bits == 4

    def test_bfpp16_layout(self):
        """BFPP16 自动生成 BitLayout"""
        layout = get_bit_layout("bfpp16")
        assert layout.name == "bfpp16"
        assert layout.precision_bits == 8

    def test_custom_format_layout(self):
        """自定义格式生成 BitLayout"""
        register_block_format(BlockFormatSpec(
            name="test_layout_fmt",
            block_size=8,
            mantissa_bits=6,
            quantize_fn=_dummy_quantize,
            dequantize_fn=_dummy_dequantize,
            storage_dtype=np.int8,
        ))
        layout = get_bit_layout("test_layout_fmt")
        assert layout.total_bits == 8
        assert layout.precision_bits == 6
        assert layout.name == "test_layout_fmt"

    def test_unknown_format_raises(self):
        """未注册格式 get_bit_layout 应报错"""
        with pytest.raises(KeyError):
            get_bit_layout("nonexistent")


class TestLoadIntegration:
    """load() 通过 registry 加载 block format"""

    def test_load_bfpp8(self, tmp_workspace):
        """BFPP8 load roundtrip"""
        from aidevtools.formats.base import load, save
        from aidevtools.formats.quantize import quantize

        data = np.random.randn(64).astype(np.float32) * 0.5
        packed, meta = quantize(data, "bfpp8")
        path = str(tmp_workspace / "test.bfpp8.bin")
        save(path, packed, fmt="raw")
        loaded = load(path, shape=(64,))
        assert loaded.dtype == np.float32
        assert loaded.shape == (64,)

    def test_load_custom_format(self, tmp_workspace):
        """自定义格式 load"""
        register_block_format(BlockFormatSpec(
            name="test_load_fmt",
            block_size=4,
            mantissa_bits=8,
            quantize_fn=_dummy_quantize,
            dequantize_fn=_dummy_dequantize,
        ))

        from aidevtools.formats.base import load, save
        from aidevtools.formats.quantize import quantize

        data = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        packed, meta = quantize(data, "test_load_fmt")
        path = str(tmp_workspace / "test.bin")
        save(path, packed, fmt="raw")
        loaded = load(path, qtype="test_load_fmt", shape=(4,))
        assert loaded.dtype == np.float32
        assert loaded.shape == (4,)
