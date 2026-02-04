"""
前端模块测试
"""

import numpy as np
import pytest

from aidevtools.frontend import (
    DataGenerator,
    DistType,
    DType,
    OpContext,
    Tensor,
    TensorMeta,
)


class TestDType:
    """数据类型测试"""

    def test_from_str(self):
        """字符串转换"""
        assert DType.from_str("fp32") == DType.FP32
        assert DType.from_str("float32") == DType.FP32
        assert DType.from_str("bfp16") == DType.BFP16
        assert DType.from_str("GFP16") == DType.GFP16  # 大写

    def test_from_str_unknown(self):
        """未知类型默认 FP32"""
        assert DType.from_str("unknown") == DType.FP32


class TestTensor:
    """Tensor 测试"""

    def test_from_numpy(self):
        """从 numpy 创建"""
        data = np.array([1.0, 2.0, 3.0])
        t = Tensor.from_numpy(data, name="test", dtype=DType.BFP16)

        assert t.shape == (3,)
        assert t.name == "test"
        assert t.dtype == DType.BFP16
        np.testing.assert_array_equal(t.numpy(), data.astype(np.float32))

    def test_empty(self):
        """创建空 Tensor"""
        t = Tensor.empty((2, 3), dtype=DType.FP16)
        assert t.shape == (2, 3)
        assert t.dtype == DType.FP16

    def test_zeros(self):
        """创建全零 Tensor"""
        t = Tensor.zeros((2, 3))
        assert t.shape == (2, 3)
        np.testing.assert_array_equal(t.numpy(), np.zeros((2, 3)))

    def test_save_load(self, tmp_path):
        """保存和加载"""
        data = np.random.randn(3, 4).astype(np.float32)
        t = Tensor.from_numpy(data, name="test")

        # 保存
        path = tmp_path / "tensor"
        t.save(path)

        # 加载
        loaded = Tensor.load(path)
        np.testing.assert_array_almost_equal(loaded.numpy(), data)


class TestDataGenerator:
    """数据生成器测试"""

    def test_seed_reproducibility(self):
        """种子可重复性"""
        gen1 = DataGenerator(seed=42)
        gen2 = DataGenerator(seed=42)

        t1 = gen1.gen_input((10, 10), dist="normal")
        t2 = gen2.gen_input((10, 10), dist="normal")

        np.testing.assert_array_equal(t1.numpy(), t2.numpy())

    def test_gen_input_normal(self):
        """正态分布输入"""
        gen = DataGenerator(seed=42)
        t = gen.gen_input((100, 100), dtype="bfp16", dist="normal")

        assert t.shape == (100, 100)
        assert t.dtype == DType.BFP16
        # 正态分布均值应接近 0
        assert abs(t.numpy().mean()) < 0.1

    def test_gen_input_uniform(self):
        """均匀分布输入"""
        gen = DataGenerator(seed=42)
        t = gen.gen_input((100, 100), dist="uniform", low=-2, high=2)

        data = t.numpy()
        assert data.min() >= -2
        assert data.max() <= 2

    def test_gen_weight_xavier(self):
        """Xavier 初始化权重"""
        gen = DataGenerator(seed=42)
        t = gen.gen_weight((64, 128), init="xavier")

        # Xavier 初始化范围检查
        limit = np.sqrt(6.0 / (64 + 128))
        data = t.numpy()
        assert data.min() >= -limit - 0.01
        assert data.max() <= limit + 0.01

    def test_gen_weight_kaiming(self):
        """Kaiming 初始化权重"""
        gen = DataGenerator(seed=42)
        t = gen.gen_weight((64, 128), init="kaiming")

        # Kaiming 初始化标准差检查
        expected_std = np.sqrt(2.0 / 128)
        actual_std = t.numpy().std()
        assert abs(actual_std - expected_std) < 0.1

    def test_gen_zeros(self):
        """生成全零"""
        gen = DataGenerator()
        t = gen.gen_zeros((3, 4))
        np.testing.assert_array_equal(t.numpy(), np.zeros((3, 4)))

    def test_gen_ones(self):
        """生成全一"""
        gen = DataGenerator()
        t = gen.gen_ones((3, 4))
        np.testing.assert_array_equal(t.numpy(), np.ones((3, 4)))

    def test_auto_naming(self):
        """自动命名"""
        gen = DataGenerator(seed=42)

        t1 = gen.gen_input((2, 2))
        t2 = gen.gen_input((2, 2))
        t3 = gen.gen_weight((2, 2))

        assert t1.name == "input_0"
        assert t2.name == "input_1"
        assert t3.name == "weight_2"

    def test_reset(self):
        """重置生成器"""
        gen = DataGenerator(seed=42)

        t1 = gen.gen_input((10,))
        gen.reset()
        t2 = gen.gen_input((10,))

        np.testing.assert_array_equal(t1.numpy(), t2.numpy())


class TestOpContext:
    """OpContext 测试"""

    def test_default_values(self):
        """默认值"""
        ctx = OpContext()
        assert ctx.dtype == DType.FP32
        assert ctx.enable_gc is True
        assert ctx.gc_level == 2

    def test_custom_values(self):
        """自定义值"""
        ctx = OpContext(
            dtype=DType.BFP16,
            enable_gc=False,
            gc_level=1,
            name="test_ctx",
        )
        assert ctx.dtype == DType.BFP16
        assert ctx.enable_gc is False
        assert ctx.gc_level == 1
        assert ctx.name == "test_ctx"


class TestCompileConfig:
    """编译配置测试"""

    def test_default_values(self):
        """默认值"""
        from aidevtools.frontend import CompileConfig

        config = CompileConfig()
        assert config.output_dir == "./build"
        assert config.golden_dir == "./golden"
        assert config.target == "dut"
        assert config.optimize == 2
        assert config.verbose is False

    def test_custom_values(self):
        """自定义值"""
        from aidevtools.frontend import CompileConfig

        config = CompileConfig(
            output_dir="/tmp/build",
            golden_dir="/tmp/golden",
            target="sim",
            optimize=3,
            verbose=True,
            py2c_version="1.0.0",
            c2dut_version="2.0.0",
        )
        assert config.output_dir == "/tmp/build"
        assert config.golden_dir == "/tmp/golden"
        assert config.target == "sim"
        assert config.optimize == 3
        assert config.verbose is True
        assert config.py2c_version == "1.0.0"
        assert config.c2dut_version == "2.0.0"


class TestCompiler:
    """编译器封装测试"""

    def test_compiler_init(self):
        """编译器初始化"""
        from aidevtools.frontend.compile import Compiler

        compiler = Compiler(
            py2c_version="1.0.0",
            c2dut_version="2.0.0",
            verbose=True,
        )
        assert compiler.py2c_version == "1.0.0"
        assert compiler.c2dut_version == "2.0.0"
        assert compiler.verbose is True

    def test_compile_error(self):
        """编译错误异常"""
        from aidevtools.frontend.compile import CompileError

        with pytest.raises(CompileError) as exc_info:
            raise CompileError("Test error")
        assert "Test error" in str(exc_info.value)

    def test_compile_python_source_not_found(self, tmp_path):
        """Python 源文件不存在"""
        from aidevtools.frontend.compile import Compiler, CompileError

        compiler = Compiler()
        source = tmp_path / "nonexistent.py"
        output = tmp_path / "output.bin"

        with pytest.raises(CompileError) as exc_info:
            compiler.compile_python(source, output)
        assert "not found" in str(exc_info.value)

    def test_compile_c_source_not_found(self, tmp_path):
        """C 源文件不存在"""
        from aidevtools.frontend.compile import Compiler, CompileError

        compiler = Compiler()
        source = tmp_path / "nonexistent.c"
        output = tmp_path / "output.bin"

        with pytest.raises(CompileError) as exc_info:
            compiler.compile_c(source, output)
        assert "not found" in str(exc_info.value)

    def test_compile_to_dut_unsupported_extension(self, tmp_path):
        """不支持的文件扩展名"""
        from aidevtools.frontend.compile import compile_to_dut, CompileError

        source = tmp_path / "test.txt"
        source.write_text("content")
        output = tmp_path / "output.bin"

        with pytest.raises(CompileError) as exc_info:
            compile_to_dut(source, output)
        assert "Unsupported" in str(exc_info.value)


class TestTensorMeta:
    """TensorMeta 测试"""

    def test_tensor_meta_creation(self):
        """TensorMeta 创建"""
        meta = TensorMeta(
            shape=(2, 3, 4),
            dtype=DType.BFP16,
            name="test_tensor",
            qtype="int8",
            scale=0.1,
            zero_point=128,
        )
        assert meta.shape == (2, 3, 4)
        assert meta.dtype == DType.BFP16
        assert meta.name == "test_tensor"
        assert meta.qtype == "int8"
        assert meta.scale == 0.1
        assert meta.zero_point == 128

    def test_tensor_meta_defaults(self):
        """TensorMeta 默认值"""
        meta = TensorMeta(shape=(10,))
        assert meta.dtype == DType.FP32
        assert meta.name == ""
        assert meta.qtype is None
        assert meta.scale is None
        assert meta.zero_point is None


class TestTensorExtended:
    """扩展 Tensor 测试"""

    def test_tensor_auto_meta_shape(self):
        """Tensor 自动设置 meta shape"""
        data = np.random.randn(3, 4).astype(np.float32)
        t = Tensor(data=data)
        assert t.meta.shape == (3, 4)

    def test_tensor_with_quant_data(self, tmp_path):
        """带量化数据的 Tensor"""
        data = np.random.randn(3, 4).astype(np.float32)
        quant_data = b"\x00\x01\x02\x03"
        t = Tensor(
            data=data,
            quant_data=quant_data,
            meta=TensorMeta(shape=(3, 4), dtype=DType.INT8),
        )

        # 保存
        path = tmp_path / "tensor"
        t.save(path)

        # 验证量化数据文件存在
        assert (tmp_path / "tensor.bin").exists()

        # 加载
        loaded = Tensor.load(path)
        assert loaded.quant_data == quant_data


class TestDataGeneratorExtended:
    """扩展数据生成器测试"""

    def test_gen_input_zeros_dist(self):
        """生成零分布输入"""
        gen = DataGenerator(seed=42)
        t = gen.gen_input((3, 4), dist="zeros")
        np.testing.assert_array_equal(t.numpy(), np.zeros((3, 4)))

    def test_gen_input_ones_dist(self):
        """生成一分布输入"""
        gen = DataGenerator(seed=42)
        t = gen.gen_input((3, 4), dist="ones")
        np.testing.assert_array_equal(t.numpy(), np.ones((3, 4)))

    def test_gen_weight_unknown_init(self):
        """未知初始化方法默认 xavier"""
        gen = DataGenerator(seed=42)
        t = gen.gen_weight((64, 128), init="unknown_init")

        # 应使用 xavier 初始化
        limit = np.sqrt(6.0 / (64 + 128))
        data = t.numpy()
        assert data.min() >= -limit - 0.01
        assert data.max() <= limit + 0.01

    def test_gen_input_with_custom_name(self):
        """自定义名称输入"""
        gen = DataGenerator(seed=42)
        t = gen.gen_input((2, 2), name="custom_input")
        assert t.name == "custom_input"

    def test_gen_weight_with_custom_name(self):
        """自定义名称权重"""
        gen = DataGenerator(seed=42)
        t = gen.gen_weight((2, 2), name="custom_weight")
        assert t.name == "custom_weight"

    def test_reset_with_new_seed(self):
        """使用新种子重置"""
        gen = DataGenerator(seed=42)
        t1 = gen.gen_input((10,))

        gen.reset(seed=123)
        t2 = gen.gen_input((10,))

        # 不同种子应产生不同数据
        assert not np.array_equal(t1.numpy(), t2.numpy())

    def test_gen_weight_with_string_dtype(self):
        """字符串 dtype 的权重生成"""
        gen = DataGenerator(seed=42)
        t = gen.gen_weight((3, 4), dtype="bfp16")
        assert t.dtype == DType.BFP16

    def test_gen_zeros_with_string_dtype(self):
        """字符串 dtype 的全零生成"""
        gen = DataGenerator(seed=42)
        t = gen.gen_zeros((3, 4), dtype="fp16")
        assert t.dtype == DType.FP16
        np.testing.assert_array_equal(t.numpy(), np.zeros((3, 4)))

    def test_gen_ones_with_string_dtype(self):
        """字符串 dtype 的全一生成"""
        gen = DataGenerator(seed=42)
        t = gen.gen_ones((3, 4), dtype="bf16")
        assert t.dtype == DType.BF16
        np.testing.assert_array_equal(t.numpy(), np.ones((3, 4)))

    def test_gen_data_unknown_dist(self):
        """未知分布类型"""
        gen = DataGenerator(seed=42)

        # 使用一个无效的分布类型会引发 ValueError
        with pytest.raises(ValueError) as exc_info:
            gen._gen_data((3, 4), "invalid_dist")
        assert "Unknown distribution" in str(exc_info.value)


class TestDTypeExtended:
    """扩展数据类型测试"""

    def test_all_dtype_values(self):
        """所有数据类型值"""
        assert DType.FP32.value == "fp32"
        assert DType.FP16.value == "fp16"
        assert DType.BF16.value == "bf16"
        assert DType.BFP16.value == "bfp16"
        assert DType.BFP8.value == "bfp8"
        assert DType.GFP16.value == "gfp16"
        assert DType.GFP8.value == "gfp8"
        assert DType.INT8.value == "int8"

    def test_from_str_all_aliases(self):
        """所有字符串别名"""
        assert DType.from_str("fp32") == DType.FP32
        assert DType.from_str("float32") == DType.FP32
        assert DType.from_str("fp16") == DType.FP16
        assert DType.from_str("float16") == DType.FP16
        assert DType.from_str("bf16") == DType.BF16
        assert DType.from_str("bfloat16") == DType.BF16
        assert DType.from_str("bfp16") == DType.BFP16
        assert DType.from_str("bfp8") == DType.BFP8
        assert DType.from_str("gfp16") == DType.GFP16
        assert DType.from_str("gfp8") == DType.GFP8
        assert DType.from_str("int8") == DType.INT8


class TestDistType:
    """分布类型测试"""

    def test_all_dist_values(self):
        """所有分布类型值"""
        assert DistType.NORMAL.value == "normal"
        assert DistType.UNIFORM.value == "uniform"
        assert DistType.ZEROS.value == "zeros"
        assert DistType.ONES.value == "ones"
        assert DistType.XAVIER.value == "xavier"
        assert DistType.KAIMING.value == "kaiming"


class TestCompilerWithMocks:
    """使用模拟的编译器测试"""

    def test_compile_python_success(self, tmp_path, monkeypatch):
        """Python 编译成功"""
        from unittest.mock import MagicMock, patch
        from aidevtools.frontend.compile import Compiler

        # 创建源文件
        source = tmp_path / "model.py"
        source.write_text("# test model")
        output = tmp_path / "build" / "model.bin"

        # 模拟 get_compiler 返回路径
        monkeypatch.setattr(
            "aidevtools.frontend.compile.get_compiler",
            lambda name, version=None: f"/mock/{name}"
        )

        # 模拟 subprocess.run
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "OK"
        mock_result.stderr = ""

        with patch("subprocess.run", return_value=mock_result) as mock_run:
            compiler = Compiler(verbose=True)
            result = compiler.compile_python(source, output)

            assert result["success"] is True
            assert mock_run.call_count == 2  # py2c + c2dut

    def test_compile_c_success(self, tmp_path, monkeypatch):
        """C 编译成功"""
        from unittest.mock import MagicMock, patch
        from aidevtools.frontend.compile import Compiler

        # 创建源文件
        source = tmp_path / "model.c"
        source.write_text("// test model")
        output = tmp_path / "build" / "model.bin"

        # 模拟 get_compiler
        monkeypatch.setattr(
            "aidevtools.frontend.compile.get_compiler",
            lambda name, version=None: f"/mock/{name}"
        )

        # 模拟 subprocess.run
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "OK"
        mock_result.stderr = ""

        with patch("subprocess.run", return_value=mock_result) as mock_run:
            compiler = Compiler()
            result = compiler.compile_c(source, output)

            assert result["success"] is True
            assert mock_run.call_count == 1  # 只有 c2dut

    def test_compile_python_with_options(self, tmp_path, monkeypatch):
        """带选项的 Python 编译"""
        from unittest.mock import MagicMock, patch
        from aidevtools.frontend.compile import Compiler

        source = tmp_path / "model.py"
        source.write_text("# test")
        output = tmp_path / "model.bin"

        monkeypatch.setattr(
            "aidevtools.frontend.compile.get_compiler",
            lambda name, version=None: f"/mock/{name}"
        )

        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = ""
        mock_result.stderr = ""

        with patch("subprocess.run", return_value=mock_result):
            compiler = Compiler()
            options = {
                "py2c": {"opt1": "value1"},
                "c2dut": {"opt2": "value2"},
            }
            result = compiler.compile_python(source, output, options=options)
            assert result["success"] is True

    def test_compile_c_with_options(self, tmp_path, monkeypatch):
        """带选项的 C 编译"""
        from unittest.mock import MagicMock, patch
        from aidevtools.frontend.compile import Compiler

        source = tmp_path / "model.c"
        source.write_text("// test")
        output = tmp_path / "model.bin"

        monkeypatch.setattr(
            "aidevtools.frontend.compile.get_compiler",
            lambda name, version=None: f"/mock/{name}"
        )

        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = ""
        mock_result.stderr = ""

        with patch("subprocess.run", return_value=mock_result):
            compiler = Compiler()
            options = {"c2dut": {"target": "sim"}}
            result = compiler.compile_c(source, output, options=options)
            assert result["success"] is True

    def test_compile_failure(self, tmp_path, monkeypatch):
        """编译失败"""
        from unittest.mock import MagicMock, patch
        from aidevtools.frontend.compile import Compiler, CompileError

        source = tmp_path / "model.c"
        source.write_text("// bad code")
        output = tmp_path / "model.bin"

        monkeypatch.setattr(
            "aidevtools.frontend.compile.get_compiler",
            lambda name, version=None: f"/mock/{name}"
        )

        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stdout = ""
        mock_result.stderr = "Error: syntax error"

        with patch("subprocess.run", return_value=mock_result):
            compiler = Compiler()
            with pytest.raises(CompileError) as exc_info:
                compiler.compile_c(source, output)
            assert "syntax error" in str(exc_info.value)

    def test_compile_timeout(self, tmp_path, monkeypatch):
        """编译超时"""
        import subprocess
        from unittest.mock import patch
        from aidevtools.frontend.compile import Compiler, CompileError

        source = tmp_path / "model.c"
        source.write_text("// slow code")
        output = tmp_path / "model.bin"

        monkeypatch.setattr(
            "aidevtools.frontend.compile.get_compiler",
            lambda name, version=None: f"/mock/{name}"
        )

        with patch("subprocess.run", side_effect=subprocess.TimeoutExpired("cmd", 300)):
            compiler = Compiler()
            with pytest.raises(CompileError) as exc_info:
                compiler.compile_c(source, output)
            assert "timed out" in str(exc_info.value)

    def test_compile_not_found(self, tmp_path, monkeypatch):
        """编译器不存在"""
        from unittest.mock import patch
        from aidevtools.frontend.compile import Compiler, CompileError

        source = tmp_path / "model.c"
        source.write_text("// test")
        output = tmp_path / "model.bin"

        monkeypatch.setattr(
            "aidevtools.frontend.compile.get_compiler",
            lambda name, version=None: "/nonexistent/compiler"
        )

        with patch("subprocess.run", side_effect=FileNotFoundError()):
            compiler = Compiler()
            with pytest.raises(CompileError) as exc_info:
                compiler.compile_c(source, output)
            assert "not found" in str(exc_info.value)

    def test_compile_to_dut_python(self, tmp_path, monkeypatch):
        """便捷函数编译 Python"""
        from unittest.mock import MagicMock, patch
        from aidevtools.frontend.compile import compile_to_dut

        source = tmp_path / "model.py"
        source.write_text("# test")
        output = tmp_path / "model.bin"

        monkeypatch.setattr(
            "aidevtools.frontend.compile.get_compiler",
            lambda name, version=None: f"/mock/{name}"
        )

        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = ""
        mock_result.stderr = ""

        with patch("subprocess.run", return_value=mock_result):
            result = compile_to_dut(source, output)
            assert result["success"] is True

    def test_compile_to_dut_c(self, tmp_path, monkeypatch):
        """便捷函数编译 C"""
        from unittest.mock import MagicMock, patch
        from aidevtools.frontend.compile import compile_to_dut

        source = tmp_path / "model.c"
        source.write_text("// test")
        output = tmp_path / "model.bin"

        monkeypatch.setattr(
            "aidevtools.frontend.compile.get_compiler",
            lambda name, version=None: f"/mock/{name}"
        )

        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = ""
        mock_result.stderr = ""

        with patch("subprocess.run", return_value=mock_result):
            result = compile_to_dut(source, output)
            assert result["success"] is True

    def test_compile_to_dut_cpp(self, tmp_path, monkeypatch):
        """便捷函数编译 C++"""
        from unittest.mock import MagicMock, patch
        from aidevtools.frontend.compile import compile_to_dut

        source = tmp_path / "model.cpp"
        source.write_text("// test")
        output = tmp_path / "model.bin"

        monkeypatch.setattr(
            "aidevtools.frontend.compile.get_compiler",
            lambda name, version=None: f"/mock/{name}"
        )

        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = ""
        mock_result.stderr = ""

        with patch("subprocess.run", return_value=mock_result):
            result = compile_to_dut(source, output)
            assert result["success"] is True

    def test_compiler_paths_cached(self, monkeypatch):
        """编译器路径缓存"""
        from aidevtools.frontend.compile import Compiler

        call_count = {"py2c": 0, "c2dut": 0}

        def mock_get_compiler(name, version=None):
            call_count[name] += 1
            return f"/mock/{name}"

        monkeypatch.setattr(
            "aidevtools.frontend.compile.get_compiler",
            mock_get_compiler
        )

        compiler = Compiler()

        # 首次访问
        _ = compiler.py2c_path
        _ = compiler.c2dut_path

        # 再次访问 (应该使用缓存)
        _ = compiler.py2c_path
        _ = compiler.c2dut_path

        assert call_count["py2c"] == 1
        assert call_count["c2dut"] == 1
