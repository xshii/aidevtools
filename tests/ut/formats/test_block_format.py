"""Block Format 注册框架单元测试"""
import numpy as np
import pytest

from aidevtools.formats.block_format import (
    BlockFormatSpec,
    DecodeResult,
    FormatInfo,
    register_block_format,
    get_block_format,
    get_format_info,
    is_block_format,
    list_block_formats,
    decode,
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
            info=FormatInfo(name="test_dummy8", description="Test dummy format"),
            block_size=4,
            quantize_fn=_dummy_quantize,
            dequantize_fn=_dummy_dequantize,
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
        assert spec.storage_dtype == np.int8
        assert spec.bytes_per_block == 33

        spec16 = get_block_format("bfpp16")
        assert spec16.block_size == 16
        assert spec16.bytes_per_block == 17

        spec4 = get_block_format("bfpp4")
        assert spec4.block_size == 64
        assert spec4.bytes_per_block == 65


class TestQuantizeDequantize:
    """通过 registry 自动注册的 quantize/dequantize 测试"""

    def test_quantize_via_registry(self):
        """注册后可通过 quantize() 调用"""
        register_block_format(BlockFormatSpec(
            info=FormatInfo(name="test_q_fmt"),
            block_size=4,
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
            info=FormatInfo(name="test_dq_fmt"),
            block_size=4,
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
    """bit_layout 符号化布局测试 (索引表达)"""

    def test_bfpp8_bit_layout(self):
        """BFPP8 (mantissa_bits=4): E*8 S0 M0*3 S1 M1*3 ... S31 M31*3"""
        spec = get_block_format("bfpp8")
        assert spec.bit_layout.startswith("E*8 S0 M0*3 S1 M1*3")
        assert spec.bit_layout.endswith("S31 M31*3")

    def test_bfpp16_bit_layout(self):
        """BFPP16 (mantissa_bits=8): E*8 S0 M0*7 ... S15 M15*7"""
        spec = get_block_format("bfpp16")
        assert spec.bit_layout.startswith("E*8 S0 M0*7")
        assert spec.bit_layout.endswith("S15 M15*7")

    def test_bfpp4_bit_layout(self):
        """BFPP4 (mantissa_bits=2): E*8 S0 M0*1 ... S63 M63*1"""
        spec = get_block_format("bfpp4")
        assert spec.bit_layout.startswith("E*8 S0 M0*1")
        assert spec.bit_layout.endswith("S63 M63*1")

    def test_gfloat16_bit_layout(self):
        """GFloat16: S0 E0*8 M0*7"""
        spec = get_block_format("gfloat16")
        assert spec.bit_layout == "S0 E0*8 M0*7"

    def test_gfloat8_bit_layout(self):
        """GFloat8: S0 E0*4 M0*3"""
        spec = get_block_format("gfloat8")
        assert spec.bit_layout == "S0 E0*4 M0*3"

    def test_gfloat4_bit_layout(self):
        """GFloat4: S0 E0*2 M0*1"""
        spec = get_block_format("gfloat4")
        assert spec.bit_layout == "S0 E0*2 M0*1"

    def test_float32_bit_layout(self):
        """float32: S0 E0*8 M0*23"""
        spec = get_block_format("float32")
        assert spec.bit_layout == "S0 E0*8 M0*23"

    def test_float16_bit_layout(self):
        """float16: S0 E0*5 M0*10"""
        spec = get_block_format("float16")
        assert spec.bit_layout == "S0 E0*5 M0*10"

    def test_bfloat16_bit_layout(self):
        """bfloat16: S0 E0*8 M0*7"""
        spec = get_block_format("bfloat16")
        assert spec.bit_layout == "S0 E0*8 M0*7"

    def test_custom_format_default_empty(self):
        """自定义格式默认 bit_layout 为空"""
        register_block_format(BlockFormatSpec(
            info=FormatInfo(name="test_layout_fmt"),
            block_size=8,
            quantize_fn=_dummy_quantize,
            dequantize_fn=_dummy_dequantize,
        ))
        spec = get_block_format("test_layout_fmt")
        assert spec.bit_layout == ""


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
            info=FormatInfo(name="test_load_fmt", bytes_per_block=4),
            block_size=4,
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


class TestDecodeResult:
    """DecodeResult 接口测试"""

    def test_bfpp8_decode_returns_decode_result(self):
        """BFPP8 decode() 返回 DecodeResult"""
        from aidevtools.formats.quantize import quantize
        data = np.random.randn(64).astype(np.float32) * 0.5
        packed, meta = quantize(data, "bfpp8")
        result = decode(packed, "bfpp8", meta)
        assert isinstance(result, DecodeResult)
        assert result.values.dtype == np.float32
        assert result.sign.dtype == np.uint8
        assert len(result.values) == 64
        assert len(result.sign) == 64
        assert len(result.mantissa) == 64
        # exponent 应被扩充到 per-element
        assert len(result.exponent) == 64

    def test_bfpp8_decode_shared_exp_expansion(self):
        """decode() 自动扩充共享指数"""
        from aidevtools.formats.quantize import quantize
        data = np.random.randn(64).astype(np.float32) * 0.5
        packed, meta = quantize(data, "bfpp8")
        # 直接调用 dequantize_fn 得到未扩充的结果
        spec = get_block_format("bfpp8")
        raw_result = spec.dequantize_fn(packed, meta)
        assert len(raw_result.exponent) == meta["num_blocks"]  # 未扩充
        # decode() 自动扩充
        result = decode(packed, "bfpp8", meta)
        assert len(result.exponent) == 64

    def test_decode_sign_values(self):
        """sign 字段正确：正数=0，负数=1"""
        from aidevtools.formats.quantize import quantize
        data = np.array([1.0, -2.0, 3.0, -4.0] * 8, dtype=np.float32)
        packed, meta = quantize(data, "bfpp16")
        result = decode(packed, "bfpp16", meta)
        # 正数的 sign 应为 0，负数的 sign 应为 1
        for i in range(len(data)):
            if data[i] >= 0:
                assert result.sign[i] == 0
            else:
                assert result.sign[i] == 1


class TestIEEERegistration:
    """IEEE 标准浮点格式统一注册测试"""

    def test_float32_registered(self):
        """float32 已注册"""
        assert is_block_format("float32")
        spec = get_block_format("float32")
        assert spec.block_size == 1
        assert spec.storage_dtype == np.float32
        assert spec.bytes_per_block == 4

    def test_float16_registered(self):
        """float16 已注册"""
        assert is_block_format("float16")
        spec = get_block_format("float16")
        assert spec.block_size == 1
        assert spec.storage_dtype == np.float16
        assert spec.bytes_per_block == 2

    def test_bfloat16_registered(self):
        """bfloat16 已注册"""
        assert is_block_format("bfloat16")
        spec = get_block_format("bfloat16")
        assert spec.block_size == 1
        assert spec.storage_dtype == np.uint16
        assert spec.bytes_per_block == 2

    def test_float32_roundtrip(self):
        """float32 quantize -> dequantize roundtrip (identity)"""
        from aidevtools.formats.quantize import quantize, dequantize
        data = np.array([1.0, -2.0, 0.5, 100.0], dtype=np.float32)
        packed, meta = quantize(data, "float32")
        restored = dequantize(packed, "float32", meta)
        assert restored.dtype == np.float32
        np.testing.assert_array_equal(data, restored)

    def test_float16_roundtrip(self):
        """float16 quantize -> dequantize roundtrip"""
        from aidevtools.formats.quantize import quantize, dequantize
        data = np.array([1.0, -2.0, 0.5, 100.0], dtype=np.float32)
        packed, meta = quantize(data, "float16")
        restored = dequantize(packed, "float16", meta)
        assert restored.dtype == np.float32
        assert np.allclose(data, restored, rtol=1e-3)

    def test_bfloat16_roundtrip(self):
        """bfloat16 quantize -> dequantize roundtrip"""
        from aidevtools.formats.quantize import quantize, dequantize
        data = np.array([1.0, -2.0, 0.5, 100.0], dtype=np.float32)
        packed, meta = quantize(data, "bfloat16")
        restored = dequantize(packed, "bfloat16", meta)
        assert restored.dtype == np.float32
        assert np.allclose(data, restored, rtol=1e-2)

    def test_float32_decode(self):
        """float32 decode() 返回 DecodeResult"""
        from aidevtools.formats.quantize import quantize
        data = np.array([1.0, -2.0, 0.0], dtype=np.float32)
        packed, meta = quantize(data, "float32")
        result = decode(packed, "float32", meta)
        assert isinstance(result, DecodeResult)
        assert len(result.values) == 3
        assert result.sign[0] == 0   # 1.0 正数
        assert result.sign[1] == 1   # -2.0 负数

    def test_float32_load(self, tmp_workspace):
        """float32 通过 load() 加载"""
        from aidevtools.formats.base import load, save
        data = np.array([1.0, -2.0, 0.5, 100.0], dtype=np.float32)
        path = str(tmp_workspace / "test.float32.bin")
        save(path, data, fmt="raw")
        loaded = load(path, shape=(4,))
        assert loaded.dtype == np.float32
        np.testing.assert_array_equal(data, loaded)

    def test_float16_load(self, tmp_workspace):
        """float16 通过 load() 加载"""
        from aidevtools.formats.base import load, save
        data = np.array([1.0, -2.0, 0.5, 100.0], dtype=np.float32)
        fp16_data = data.astype(np.float16)
        path = str(tmp_workspace / "test.float16.bin")
        save(path, fp16_data, fmt="raw")
        loaded = load(path, shape=(4,))
        assert loaded.dtype == np.float32
        assert np.allclose(data, loaded, rtol=1e-3)


class TestGFloatRegistration:
    """GFloat 统一注册测试"""

    def test_gfloat16_registered(self):
        """gfloat16 已通过 register_block_format 注册"""
        assert is_block_format("gfloat16")
        spec = get_block_format("gfloat16")
        assert spec.block_size == 1
        assert spec.storage_dtype == np.uint16
        assert spec.bytes_per_block == 2  # 1 element * uint16

    def test_gfloat8_registered(self):
        """gfloat8 已注册"""
        assert is_block_format("gfloat8")
        spec = get_block_format("gfloat8")
        assert spec.block_size == 1
        assert spec.storage_dtype == np.uint8
        assert spec.bytes_per_block == 1  # 1 element * uint8

    def test_gfloat4_registered(self):
        """gfloat4 已注册"""
        assert is_block_format("gfloat4")
        spec = get_block_format("gfloat4")
        assert spec.block_size == 1
        assert spec.storage_dtype == np.uint8

    def test_gfp_aliases_registered(self):
        """gfp16/gfp8/gfp4 别名已注册"""
        assert is_block_format("gfp16")
        assert is_block_format("gfp8")
        assert is_block_format("gfp4")

    def test_gfloat16_roundtrip(self):
        """GFloat16 quantize -> dequantize roundtrip"""
        from aidevtools.formats.quantize import quantize, dequantize
        data = np.array([1.0, -2.0, 0.5, 100.0], dtype=np.float32)
        packed, meta = quantize(data, "gfloat16")
        restored = dequantize(packed, "gfloat16", meta)
        assert restored.dtype == np.float32
        assert np.allclose(data, restored, rtol=1e-3)

    def test_gfloat8_roundtrip(self):
        """GFloat8 quantize -> dequantize roundtrip"""
        from aidevtools.formats.quantize import quantize, dequantize
        data = np.array([1.0, -2.0, 0.5, 64.0], dtype=np.float32)
        packed, meta = quantize(data, "gfloat8")
        restored = dequantize(packed, "gfloat8", meta)
        assert restored.dtype == np.float32
        assert restored.shape == (4,)

    def test_gfloat16_decode(self):
        """GFloat16 decode() 返回 DecodeResult"""
        from aidevtools.formats.quantize import quantize
        data = np.array([1.0, -2.0, 0.5, 100.0], dtype=np.float32)
        packed, meta = quantize(data, "gfloat16")
        result = decode(packed, "gfloat16", meta)
        assert isinstance(result, DecodeResult)
        assert len(result.values) == 4
        assert len(result.sign) == 4
        assert len(result.exponent) == 4
        assert len(result.mantissa) == 4
        # -2.0 的 sign 应为 1
        assert result.sign[1] == 1
        # 1.0 的 sign 应为 0
        assert result.sign[0] == 0

    def test_gfloat16_load(self, tmp_workspace):
        """GFloat16 通过 load() 加载"""
        from aidevtools.formats.base import load, save
        from aidevtools.formats.quantize import quantize
        data = np.array([1.0, -2.0, 0.5, 100.0], dtype=np.float32)
        packed, meta = quantize(data, "gfloat16")
        path = str(tmp_workspace / "test.gfloat16.bin")
        save(path, packed, fmt="raw")
        loaded = load(path, shape=(4,))
        assert loaded.dtype == np.float32
        assert loaded.shape == (4,)
        assert np.allclose(data, loaded, rtol=1e-3)
