"""格式模块测试"""
import pytest
import numpy as np

from aidevtools.formats.base import load, load_dir, save

class TestRawFormat:
    """Raw 格式测试"""

    def test_save_load(self, tmp_workspace, sample_data):
        """保存和加载"""
        path = str(tmp_workspace / "test.bin")
        save(path, sample_data, fmt="raw")
        loaded = load(path, fmt="raw", dtype=np.float32, shape=sample_data.shape)
        assert np.allclose(sample_data, loaded)

    def test_load_reshape(self, tmp_workspace):
        """加载时 reshape"""
        data = np.arange(24, dtype=np.float32)
        path = str(tmp_workspace / "test.bin")
        save(path, data, fmt="raw")
        loaded = load(path, fmt="raw", dtype=np.float32, shape=(2, 3, 4))
        assert loaded.shape == (2, 3, 4)

class TestNumpyFormat:
    """Numpy 格式测试"""

    def test_npy(self, tmp_workspace, sample_data):
        """npy 格式"""
        path = str(tmp_workspace / "test.npy")
        save(path, sample_data, fmt="numpy")
        loaded = load(path, fmt="numpy")
        assert np.allclose(sample_data, loaded)

    def test_npz(self, tmp_workspace, sample_data):
        """npz 格式"""
        path = str(tmp_workspace / "test.npz")
        save(path, sample_data, fmt="numpy")
        loaded = load(path, fmt="numpy")
        assert np.allclose(sample_data, loaded)


class TestFormatEdgeCases:
    """格式边界测试"""

    def test_unknown_format_save(self, tmp_workspace, sample_data):
        """未知格式保存"""
        path = str(tmp_workspace / "test.bin")
        with pytest.raises(ValueError, match="未知格式"):
            save(path, sample_data, fmt="unknown_format")

    def test_unknown_format_load(self, tmp_workspace, sample_data):
        """未知格式加载"""
        path = str(tmp_workspace / "test.bin")
        save(path, sample_data, fmt="raw")
        with pytest.raises(ValueError, match="未知格式"):
            load(path, fmt="unknown_format")

    def test_auto_detect_npy(self, tmp_workspace, sample_data):
        """自动检测 npy 格式"""
        path = str(tmp_workspace / "test.npy")
        np.save(path, sample_data)
        # 使用 numpy 格式加载 .npy 文件
        loaded = load(path, fmt="numpy")
        assert np.allclose(sample_data, loaded)

    def test_raw_without_dtype(self, tmp_workspace, sample_data):
        """raw 格式未指定 dtype"""
        path = str(tmp_workspace / "test.bin")
        save(path, sample_data, fmt="raw")
        # 未指定 dtype 时默认为 float32
        loaded = load(path, fmt="raw", shape=sample_data.shape)
        assert loaded.dtype == np.float32


class TestLoadQtype:
    """load with qtype 测试"""

    def test_load_float32_explicit(self, tmp_workspace):
        """显式指定 qtype=float32"""
        data = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        path = str(tmp_workspace / "test.bin")
        save(path, data, fmt="raw")
        loaded = load(path, qtype="float32", shape=(3,))
        assert np.allclose(data, loaded)
        assert loaded.dtype == np.float32

    def test_load_float16_explicit(self, tmp_workspace):
        """显式指定 qtype=float16"""
        data = np.array([1.0, 2.0, 3.0], dtype=np.float16)
        path = str(tmp_workspace / "test.bin")
        data.tofile(path)
        loaded = load(path, qtype="float16", shape=(3,))
        assert loaded.dtype == np.float32
        assert np.allclose(data.astype(np.float32), loaded)

    def test_load_bfp8_roundtrip(self, tmp_workspace):
        """bfp8 量化 → 存 bin → load 回 fp32"""
        from aidevtools.formats.quantize import quantize
        data = np.random.randn(64).astype(np.float32) * 0.5
        packed, meta = quantize(data, "bfp8")
        path = str(tmp_workspace / "test.bfp8.bin")
        save(path, packed, fmt="raw")
        # 自动从文件名推断 qtype
        loaded = load(path, shape=(64,))
        assert loaded.dtype == np.float32
        assert loaded.shape == (64,)
        # 量化精度损失在合理范围内
        from aidevtools.formats.quantize import simulate_quantize
        expected = simulate_quantize(data, "bfp8")
        assert np.allclose(loaded, expected, atol=0.1)

    def test_load_bfp4_roundtrip(self, tmp_workspace):
        """bfp4 量化 → 存 bin → load 回 fp32"""
        from aidevtools.formats.quantize import quantize
        data = np.random.randn(128).astype(np.float32) * 0.5
        packed, _ = quantize(data, "bfp4")
        path = str(tmp_workspace / "tensor.bfp4.bin")
        save(path, packed, fmt="raw")
        loaded = load(path, shape=(128,))
        assert loaded.dtype == np.float32
        assert loaded.shape == (128,)

    def test_load_bfp16_roundtrip(self, tmp_workspace):
        """bfp16 量化 → 存 bin → load 回 fp32"""
        from aidevtools.formats.quantize import quantize
        data = np.random.randn(32).astype(np.float32) * 0.5
        packed, _ = quantize(data, "bfp16")
        path = str(tmp_workspace / "tensor.bfp16.bin")
        save(path, packed, fmt="raw")
        loaded = load(path, shape=(32,))
        assert loaded.dtype == np.float32
        assert loaded.shape == (32,)

    def test_infer_qtype_from_filename(self, tmp_workspace):
        """从文件名后缀推断 qtype"""
        from aidevtools.formats.base import _infer_qtype
        assert _infer_qtype("input_0.bfp8.bin") == "bfp8"
        assert _infer_qtype("weight.bfp4.bin") == "bfp4"
        assert _infer_qtype("data.bfp16.bin") == "bfp16"
        assert _infer_qtype("data.float16.bin") == "float16"
        assert _infer_qtype("data.float32.bin") == "float32"
        assert _infer_qtype("data.bin") is None
        assert _infer_qtype("data.unknown.bin") is None

    def test_load_bfp_no_shape_raises(self, tmp_workspace):
        """BFP 类型不指定 shape 应报错"""
        from aidevtools.formats.quantize import quantize
        data = np.random.randn(64).astype(np.float32)
        packed, _ = quantize(data, "bfp8")
        path = str(tmp_workspace / "test.bfp8.bin")
        save(path, packed, fmt="raw")
        with pytest.raises(ValueError, match="shape"):
            load(path)

    def test_load_explicit_qtype_overrides_filename(self, tmp_workspace):
        """显式 qtype 优先于文件名推断"""
        data = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        # 文件名暗示 bfp8，但显式指定 float32
        path = str(tmp_workspace / "test.bfp8.bin")
        save(path, data, fmt="raw")
        loaded = load(path, qtype="float32", shape=(3,))
        assert np.allclose(data, loaded)

    def test_load_multidim_shape(self, tmp_workspace):
        """多维 shape 回放"""
        from aidevtools.formats.quantize import quantize
        data = np.random.randn(2, 16, 64).astype(np.float32) * 0.5
        packed, _ = quantize(data, "bfp8")
        path = str(tmp_workspace / "tensor.bfp8.bin")
        save(path, packed, fmt="raw")
        loaded = load(path, shape=(2, 16, 64))
        assert loaded.shape == (2, 16, 64)
        assert loaded.dtype == np.float32

    def test_infer_shape_from_filename(self):
        """从文件名推断 shape"""
        from aidevtools.formats.base import _infer_shape
        assert _infer_shape("encoder_input_0_2x16x64.bfp8.bin") == (2, 16, 64)
        assert _infer_shape("enc_linear_0_weight_64x64.bfp4.bin") == (64, 64)
        assert _infer_shape("data_128.bfp8.bin") == (128,)
        assert _infer_shape("data.bfp8.bin") is None
        assert _infer_shape("data_abc.bfp8.bin") is None

    def test_load_fully_auto(self, tmp_workspace):
        """全自动: 文件名同时包含 shape 和 qtype"""
        from aidevtools.formats.quantize import quantize, simulate_quantize
        data = np.random.randn(2, 16, 64).astype(np.float32) * 0.5
        packed, _ = quantize(data, "bfp8")
        path = str(tmp_workspace / "encoder_input_0_2x16x64.bfp8.bin")
        save(path, packed, fmt="raw")
        # 无需指定 qtype 和 shape，全从文件名推断
        loaded = load(path)
        assert loaded.shape == (2, 16, 64)
        assert loaded.dtype == np.float32
        expected = simulate_quantize(data, "bfp8")
        assert np.allclose(loaded, expected, atol=0.1)


class TestExportNaming:
    """export 文件名测试"""

    def test_export_with_bm_prefix(self, tmp_workspace):
        """bm 前缀 + shape 后缀"""
        from aidevtools.datagen import DataGenerator
        gen = DataGenerator(seed=42, qtype="bfp8")
        gen.randn((2, 16, 64), name="input_0")
        result = gen.export(str(tmp_workspace), bm="encoder")
        paths = list(result.values())
        assert len(paths) == 1
        fname = paths[0].name
        # 应包含 bm 前缀、shape 后缀、qtype 后缀
        assert fname.startswith("encoder_")
        assert "2x16x64" in fname
        assert ".bfp8.bin" in fname

    def test_export_shape_in_filename(self, tmp_workspace):
        """shape 编入文件名"""
        from aidevtools.datagen import DataGenerator
        gen = DataGenerator(seed=42, qtype="bfp4")
        gen.randn((64, 64), name="weight")
        result = gen.export(str(tmp_workspace))
        paths = list(result.values())
        fname = paths[0].name
        assert "64x64" in fname
        assert ".bfp4.bin" in fname

    def test_export_load_roundtrip(self, tmp_workspace):
        """export → load 全自动 roundtrip"""
        from aidevtools.datagen import DataGenerator
        from aidevtools.formats.quantize import simulate_quantize
        gen = DataGenerator(seed=42, qtype="bfp8")
        gen.randn((32,), name="bias")
        original = gen._tensors["bias"].array.copy()
        result = gen.export(str(tmp_workspace), bm="test")
        # 全自动 load
        path = str(list(result.values())[0])
        loaded = load(path)
        expected = simulate_quantize(original, "bfp8")
        assert loaded.shape == (32,)
        assert np.allclose(loaded, expected, atol=0.1)


class TestLoadDir:
    """load_dir 自动扫描测试"""

    def test_load_dir_basic(self, tmp_workspace):
        """基本目录扫描"""
        from aidevtools.datagen import DataGenerator
        gen = DataGenerator(seed=42, qtype="bfp8")
        gen.randn((2, 16, 64), name="input_0")
        gen.randn((64, 64), name="weight_0")
        gen.export(str(tmp_workspace), bm="enc")
        tensors = load_dir(str(tmp_workspace), bm="enc")
        assert "input_0" in tensors
        assert "weight_0" in tensors
        assert tensors["input_0"].shape == (2, 16, 64)
        assert tensors["weight_0"].shape == (64, 64)
        assert tensors["input_0"].dtype == np.float32

    def test_load_dir_bm_filter(self, tmp_workspace):
        """bm 前缀过滤"""
        from aidevtools.formats.quantize import quantize
        # 两个 bm 的文件
        d1 = np.random.randn(32).astype(np.float32)
        d2 = np.random.randn(64).astype(np.float32)
        p1, _ = quantize(d1, "bfp8")
        p2, _ = quantize(d2, "bfp8")
        save(str(tmp_workspace / "enc_a_32.bfp8.bin"), p1)
        save(str(tmp_workspace / "dec_b_64.bfp8.bin"), p2)
        enc = load_dir(str(tmp_workspace), bm="enc")
        dec = load_dir(str(tmp_workspace), bm="dec")
        assert "a" in enc and "b" not in enc
        assert "b" in dec and "a" not in dec

    def test_load_dir_all(self, tmp_workspace):
        """不指定 bm，加载全部"""
        from aidevtools.formats.quantize import quantize
        d1 = np.random.randn(32).astype(np.float32)
        p1, _ = quantize(d1, "bfp8")
        save(str(tmp_workspace / "tensor_32.bfp8.bin"), p1)
        tensors = load_dir(str(tmp_workspace))
        assert "tensor" in tensors
        assert tensors["tensor"].shape == (32,)

    def test_load_dir_skips_unknown(self, tmp_workspace):
        """跳过无法推断 qtype 的文件"""
        save(str(tmp_workspace / "readme.bin"), np.zeros(10, dtype=np.float32))
        tensors = load_dir(str(tmp_workspace))
        assert len(tensors) == 0

    def test_infer_name(self):
        """从文件名提取 tensor 名"""
        from aidevtools.formats.base import _infer_name
        assert _infer_name("encoder_linear_0_weight_64x64.bfp4.bin", bm="encoder") == "linear_0_weight"
        assert _infer_name("enc_input_0_2x16x64.bfp8.bin", bm="enc") == "input_0"
        assert _infer_name("bias_32.bfp8.bin") == "bias"
        assert _infer_name("data.bfp8.bin") == "data"
