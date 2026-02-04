"""CPU Golden 单元测试"""
import pytest
import numpy as np
import subprocess
import tempfile
from pathlib import Path


# CPU Golden 可执行文件路径
CPU_GOLDEN_PATH = Path(__file__).parent.parent.parent / "aidevtools/golden/cpu_golden"


def skip_if_not_available():
    """检查 cpu_golden Python wrapper 是否可用"""
    from aidevtools.ops.cpu_golden import is_cpu_golden_available
    if not is_cpu_golden_available():
        pytest.skip("cpu_golden not available")


def skip_if_not_built():
    """检查 cpu_golden 是否已编译"""
    if not CPU_GOLDEN_PATH.exists():
        pytest.skip(f"cpu_golden not built: {CPU_GOLDEN_PATH}")


# ==================== GFloat 格式转换 ====================

def fp32_to_gfloat16(x: np.ndarray) -> np.ndarray:
    """fp32 -> gfloat16 (取高16位)"""
    bits = x.view(np.uint32)
    return (bits >> 16).astype(np.uint16)


def gfloat16_to_fp32(x: np.ndarray) -> np.ndarray:
    """gfloat16 -> fp32 (低16位补零)"""
    bits = x.astype(np.uint32) << 16
    return bits.view(np.float32)


def fp32_to_gfloat8(x: np.ndarray) -> np.ndarray:
    """fp32 -> gfloat8 (取高8位)"""
    bits = x.view(np.uint32)
    return (bits >> 24).astype(np.uint8)


def gfloat8_to_fp32(x: np.ndarray) -> np.ndarray:
    """gfloat8 -> fp32 (低24位补零)"""
    bits = x.astype(np.uint32) << 24
    return bits.view(np.float32)


def fp32_to_gfloat4(x: np.ndarray) -> np.ndarray:
    """fp32 -> gfloat4 (取高4位, packed)"""
    bits = x.view(np.uint32)
    val4 = (bits >> 28).astype(np.uint8)
    # Pack: 2个4-bit值打包到1个uint8
    size = x.size
    packed_size = (size + 1) // 2
    packed = np.zeros(packed_size, dtype=np.uint8)
    for i in range(size):
        byte_idx = i // 2
        if i % 2 == 0:
            packed[byte_idx] |= (val4.flat[i] << 4)
        else:
            packed[byte_idx] |= val4.flat[i]
    return packed


def gfloat4_to_fp32(packed: np.ndarray, size: int) -> np.ndarray:
    """gfloat4 -> fp32 (低28位补零)"""
    output = np.zeros(size, dtype=np.float32)
    for i in range(size):
        byte_idx = i // 2
        if i % 2 == 0:
            val4 = (packed[byte_idx] >> 4) & 0x0F
        else:
            val4 = packed[byte_idx] & 0x0F
        bits = np.uint32(val4) << 28
        output[i] = np.array([bits], dtype=np.uint32).view(np.float32)[0]
    return output


class TestCpuGoldenMatmul:
    """MatMul 测试"""

    def setup_method(self):
        skip_if_not_built()
        self.temp_dir = tempfile.mkdtemp()

    def test_matmul_gfp16(self):
        """MatMul gfloat16"""
        M, K, N = 4, 8, 16

        # 生成测试数据
        a = np.random.randn(M, K).astype(np.float32)
        b = np.random.randn(K, N).astype(np.float32)

        # 转换为 gfloat16
        a_gfp = fp32_to_gfloat16(a)
        b_gfp = fp32_to_gfloat16(b)

        # 保存输入
        a_path = Path(self.temp_dir) / "a.bin"
        b_path = Path(self.temp_dir) / "b.bin"
        c_path = Path(self.temp_dir) / "c.bin"

        a_gfp.tofile(a_path)
        b_gfp.tofile(b_path)

        # 调用 cpu_golden
        result = subprocess.run(
            [str(CPU_GOLDEN_PATH), "matmul", "gfp16",
             str(a_path), str(b_path), str(c_path),
             str(M), str(K), str(N)],
            capture_output=True, text=True
        )
        assert result.returncode == 0, f"Failed: {result.stderr}"

        # 读取结果
        c_gfp = np.fromfile(c_path, dtype=np.uint16)
        c = gfloat16_to_fp32(c_gfp).reshape(M, N)

        # 用量化后的输入计算参考值
        # 注意: cpu_golden 的精度模拟会量化每个中间结果，
        # 而 numpy matmul 用 fp32 计算，所以会有累积误差
        a_quant = gfloat16_to_fp32(a_gfp)
        b_quant = gfloat16_to_fp32(b_gfp)
        c_ref = np.matmul(a_quant, b_quant)

        # 验证 (增加容差以适应精度模拟的累积误差)
        assert c.shape == (M, N)
        assert np.allclose(c, c_ref, rtol=1e-1, atol=1e-1)

    def test_matmul_gfp8(self):
        """MatMul gfloat8"""
        M, K, N = 2, 4, 8

        a = np.random.randn(M, K).astype(np.float32)
        b = np.random.randn(K, N).astype(np.float32)

        a_gfp = fp32_to_gfloat8(a)
        b_gfp = fp32_to_gfloat8(b)

        a_path = Path(self.temp_dir) / "a.bin"
        b_path = Path(self.temp_dir) / "b.bin"
        c_path = Path(self.temp_dir) / "c.bin"

        a_gfp.tofile(a_path)
        b_gfp.tofile(b_path)

        result = subprocess.run(
            [str(CPU_GOLDEN_PATH), "matmul", "gfp8",
             str(a_path), str(b_path), str(c_path),
             str(M), str(K), str(N)],
            capture_output=True, text=True
        )
        assert result.returncode == 0, f"Failed: {result.stderr}"

        c_gfp = np.fromfile(c_path, dtype=np.uint8)
        c = gfloat8_to_fp32(c_gfp).reshape(M, N)

        assert c.shape == (M, N)


class TestCpuGoldenSoftmax:
    """Softmax 测试"""

    def setup_method(self):
        skip_if_not_built()
        self.temp_dir = tempfile.mkdtemp()

    def test_softmax_gfp16(self):
        """Softmax gfloat16"""
        batch, seq = 4, 16

        x = np.random.randn(batch, seq).astype(np.float32)
        x_gfp = fp32_to_gfloat16(x)

        input_path = Path(self.temp_dir) / "input.bin"
        output_path = Path(self.temp_dir) / "output.bin"

        x_gfp.tofile(input_path)

        result = subprocess.run(
            [str(CPU_GOLDEN_PATH), "softmax", "gfp16",
             str(input_path), str(output_path),
             str(batch), str(seq)],
            capture_output=True, text=True
        )
        assert result.returncode == 0, f"Failed: {result.stderr}"

        y_gfp = np.fromfile(output_path, dtype=np.uint16)
        y = gfloat16_to_fp32(y_gfp).reshape(batch, seq)

        # softmax 输出每行和应该接近 1
        row_sums = y.sum(axis=1)
        assert y.shape == (batch, seq)
        assert np.allclose(row_sums, 1.0, atol=0.05)  # gfloat16 有量化误差

    def test_softmax_gfp8(self):
        """Softmax gfloat8"""
        batch, seq = 2, 8

        x = np.random.randn(batch, seq).astype(np.float32)
        x_gfp = fp32_to_gfloat8(x)

        input_path = Path(self.temp_dir) / "input.bin"
        output_path = Path(self.temp_dir) / "output.bin"

        x_gfp.tofile(input_path)

        result = subprocess.run(
            [str(CPU_GOLDEN_PATH), "softmax", "gfp8",
             str(input_path), str(output_path),
             str(batch), str(seq)],
            capture_output=True, text=True
        )
        assert result.returncode == 0, f"Failed: {result.stderr}"

        y_gfp = np.fromfile(output_path, dtype=np.uint8)
        y = gfloat8_to_fp32(y_gfp).reshape(batch, seq)

        assert y.shape == (batch, seq)


class TestCpuGoldenLayernorm:
    """LayerNorm 测试"""

    def setup_method(self):
        skip_if_not_built()
        self.temp_dir = tempfile.mkdtemp()

    def test_layernorm_gfp16(self):
        """LayerNorm gfloat16"""
        batch, hidden = 4, 64

        x = np.random.randn(batch, hidden).astype(np.float32)
        gamma = np.ones(hidden, dtype=np.float32)
        beta = np.zeros(hidden, dtype=np.float32)

        x_gfp = fp32_to_gfloat16(x)
        gamma_gfp = fp32_to_gfloat16(gamma)
        beta_gfp = fp32_to_gfloat16(beta)

        x_path = Path(self.temp_dir) / "x.bin"
        gamma_path = Path(self.temp_dir) / "gamma.bin"
        beta_path = Path(self.temp_dir) / "beta.bin"
        y_path = Path(self.temp_dir) / "y.bin"

        x_gfp.tofile(x_path)
        gamma_gfp.tofile(gamma_path)
        beta_gfp.tofile(beta_path)

        result = subprocess.run(
            [str(CPU_GOLDEN_PATH), "layernorm", "gfp16",
             str(x_path), str(gamma_path), str(beta_path), str(y_path),
             str(batch), str(hidden)],
            capture_output=True, text=True
        )
        assert result.returncode == 0, f"Failed: {result.stderr}"

        y_gfp = np.fromfile(y_path, dtype=np.uint16)
        y = gfloat16_to_fp32(y_gfp).reshape(batch, hidden)

        # layernorm 输出每行均值应接近 0，方差接近 1
        assert y.shape == (batch, hidden)
        assert np.allclose(y.mean(axis=1), 0.0, atol=0.1)
        assert np.allclose(y.var(axis=1), 1.0, atol=0.2)

    def test_layernorm_with_scale_bias(self):
        """LayerNorm with gamma/beta"""
        batch, hidden = 2, 32

        x = np.random.randn(batch, hidden).astype(np.float32)
        gamma = np.random.randn(hidden).astype(np.float32) * 0.5 + 1.0
        beta = np.random.randn(hidden).astype(np.float32) * 0.1

        x_gfp = fp32_to_gfloat16(x)
        gamma_gfp = fp32_to_gfloat16(gamma)
        beta_gfp = fp32_to_gfloat16(beta)

        x_path = Path(self.temp_dir) / "x.bin"
        gamma_path = Path(self.temp_dir) / "gamma.bin"
        beta_path = Path(self.temp_dir) / "beta.bin"
        y_path = Path(self.temp_dir) / "y.bin"

        x_gfp.tofile(x_path)
        gamma_gfp.tofile(gamma_path)
        beta_gfp.tofile(beta_path)

        result = subprocess.run(
            [str(CPU_GOLDEN_PATH), "layernorm", "gfp16",
             str(x_path), str(gamma_path), str(beta_path), str(y_path),
             str(batch), str(hidden)],
            capture_output=True, text=True
        )
        assert result.returncode == 0, f"Failed: {result.stderr}"

        y_gfp = np.fromfile(y_path, dtype=np.uint16)
        y = gfloat16_to_fp32(y_gfp).reshape(batch, hidden)

        assert y.shape == (batch, hidden)


class TestCpuGoldenGfloat4:
    """GFloat4 格式测试"""

    def setup_method(self):
        skip_if_not_built()
        self.temp_dir = tempfile.mkdtemp()

    def test_softmax_gfp4(self):
        """Softmax gfloat4"""
        batch, seq = 2, 8

        x = np.random.randn(batch, seq).astype(np.float32)
        x_gfp = fp32_to_gfloat4(x)

        input_path = Path(self.temp_dir) / "input.bin"
        output_path = Path(self.temp_dir) / "output.bin"

        x_gfp.tofile(input_path)

        result = subprocess.run(
            [str(CPU_GOLDEN_PATH), "softmax", "gfp4",
             str(input_path), str(output_path),
             str(batch), str(seq)],
            capture_output=True, text=True
        )
        assert result.returncode == 0, f"Failed: {result.stderr}"

        y_gfp = np.fromfile(output_path, dtype=np.uint8)
        y = gfloat4_to_fp32(y_gfp, batch * seq).reshape(batch, seq)

        assert y.shape == (batch, seq)


class TestCpuGoldenCLI:
    """CLI 测试"""

    def setup_method(self):
        skip_if_not_built()

    def test_help(self):
        """--help 选项"""
        result = subprocess.run(
            [str(CPU_GOLDEN_PATH), "--help"],
            capture_output=True, text=True
        )
        assert result.returncode == 0
        assert "CPU Golden CLI" in result.stderr

    def test_unknown_op(self):
        """未知算子"""
        result = subprocess.run(
            [str(CPU_GOLDEN_PATH), "unknown_op"],
            capture_output=True, text=True
        )
        assert result.returncode == 1
        assert "unknown op" in result.stderr

    def test_missing_args(self):
        """缺少参数"""
        result = subprocess.run(
            [str(CPU_GOLDEN_PATH), "matmul", "gfp16"],
            capture_output=True, text=True
        )
        assert result.returncode == 1


# ==================== Op 类 cpu_golden 方法测试 ====================

class TestOpCpuGoldenMethod:
    """Op 类 cpu_golden 方法测试"""

    def setup_method(self):
        skip_if_not_available()
        from aidevtools.ops.cpu_golden import set_cpu_golden_dtype
        set_cpu_golden_dtype("gfp16")

    def test_matmul_cpu_golden(self):
        """MatMul.cpu_golden 方法"""
        from aidevtools.ops import _functional as F

        M, K, N = 4, 8, 16
        a = np.random.randn(M, K).astype(np.float32)
        b = np.random.randn(K, N).astype(np.float32)

        op = F.MatMul()
        c = op.cpu_golden(a, b)

        assert c.shape == (M, N)
        assert c.dtype == np.float32

    def test_softmax_cpu_golden(self):
        """Softmax.cpu_golden 方法"""
        from aidevtools.ops import _functional as F

        batch, seq = 4, 16
        x = np.random.randn(batch, seq).astype(np.float32)

        op = F.Softmax()
        y = op.cpu_golden(x)

        assert y.shape == (batch, seq)
        # softmax 输出每行和应该接近 1
        assert np.allclose(y.sum(axis=1), 1.0, atol=0.05)

    def test_softmax_1d_cpu_golden(self):
        """Softmax.cpu_golden 1D 输入"""
        from aidevtools.ops import _functional as F

        x = np.random.randn(8).astype(np.float32)

        op = F.Softmax()
        y = op.cpu_golden(x)

        assert y.shape == (8,)
        assert np.allclose(y.sum(), 1.0, atol=0.05)

    def test_layernorm_cpu_golden(self):
        """LayerNorm.cpu_golden 方法"""
        from aidevtools.ops import _functional as F

        batch, hidden = 4, 64
        x = np.random.randn(batch, hidden).astype(np.float32)
        weight = np.ones(hidden, dtype=np.float32)
        bias = np.zeros(hidden, dtype=np.float32)

        op = F.LayerNorm()
        y = op.cpu_golden(x, normalized_shape=(hidden,), weight=weight, bias=bias)

        assert y.shape == (batch, hidden)
        # layernorm 输出每行均值应接近 0
        assert np.allclose(y.mean(axis=1), 0.0, atol=0.1)

    def test_transpose_cpu_golden(self):
        """Transpose.cpu_golden 方法"""
        from aidevtools.ops import _functional as F

        d0, d1, d2, d3 = 2, 4, 8, 16
        x = np.random.randn(d0, d1, d2, d3).astype(np.float32)

        op = F.Transpose()
        y = op.cpu_golden(x)

        assert y.shape == (d0, d1, d3, d2)


class TestCppGoldenMode:
    """使用 cpp golden mode 测试"""

    def setup_method(self):
        skip_if_not_available()
        from aidevtools.ops.base import clear, set_golden_mode
        from aidevtools.ops.cpu_golden import set_cpu_golden_dtype
        clear()
        set_golden_mode("python")
        set_cpu_golden_dtype("gfp16")

    def teardown_method(self):
        from aidevtools.ops.base import set_golden_mode
        set_golden_mode("python")

    def test_use_cpp_golden_mode(self):
        """使用 cpp golden mode"""
        from aidevtools.ops import _functional as F
        from aidevtools.ops.base import set_golden_mode, clear, get_records

        set_golden_mode("cpp")
        clear()

        x = np.random.randn(2, 8).astype(np.float32)
        y = F.softmax(x)

        assert y.shape == (2, 8)
        # softmax 输出每行和应该接近 1
        assert np.allclose(y.sum(axis=1), 1.0, atol=0.1)

        records = get_records()
        assert len(records) == 1
        assert records[0]["golden"] is not None

    def test_matmul_cpp_mode(self):
        """MatMul cpp golden mode"""
        from aidevtools.ops import _functional as F
        from aidevtools.ops.base import set_golden_mode, clear, get_records

        set_golden_mode("cpp")
        clear()

        a = np.random.randn(4, 8).astype(np.float32)
        b = np.random.randn(8, 16).astype(np.float32)
        c = F.matmul(a, b)

        assert c.shape == (4, 16)

        records = get_records()
        assert len(records) == 1
        assert records[0]["golden"] is not None


# ==================== 混合精度 MatMul 测试 ====================

class TestMixedPrecisionMatmul:
    """混合精度 MatMul 测试"""

    def setup_method(self):
        skip_if_not_built()
        self.temp_dir = tempfile.mkdtemp()

    def test_matmul_mixed_gfp8_gfp4(self):
        """MatMul 混合精度: gfp8 x gfp4 -> gfp16"""
        M, K, N = 4, 8, 16

        # 生成测试数据
        a = np.random.randn(M, K).astype(np.float32)
        b = np.random.randn(K, N).astype(np.float32)

        # A 用 gfp8, B 用 gfp4
        a_gfp = fp32_to_gfloat8(a)
        b_gfp = fp32_to_gfloat4(b)

        # 保存输入
        a_path = Path(self.temp_dir) / "a.bin"
        b_path = Path(self.temp_dir) / "b.bin"
        c_path = Path(self.temp_dir) / "c.bin"

        a_gfp.tofile(a_path)
        b_gfp.tofile(b_path)

        # 调用 cpu_golden matmul_mixed
        result = subprocess.run(
            [str(CPU_GOLDEN_PATH), "matmul_mixed", "gfp8", "gfp4",
             str(a_path), str(b_path), str(c_path),
             str(M), str(K), str(N), "gfp16"],
            capture_output=True, text=True
        )
        assert result.returncode == 0, f"Failed: {result.stderr}"

        # 读取结果 (gfp16)
        c_gfp = np.fromfile(c_path, dtype=np.uint16)
        c = gfloat16_to_fp32(c_gfp).reshape(M, N)

        # 验证 shape
        assert c.shape == (M, N)
        assert c.dtype == np.float32

    def test_matmul_mixed_gfp16_gfp8(self):
        """MatMul 混合精度: gfp16 x gfp8 -> gfp16"""
        M, K, N = 2, 4, 8

        a = np.random.randn(M, K).astype(np.float32)
        b = np.random.randn(K, N).astype(np.float32)

        a_gfp = fp32_to_gfloat16(a)
        b_gfp = fp32_to_gfloat8(b)

        a_path = Path(self.temp_dir) / "a.bin"
        b_path = Path(self.temp_dir) / "b.bin"
        c_path = Path(self.temp_dir) / "c.bin"

        a_gfp.tofile(a_path)
        b_gfp.tofile(b_path)

        result = subprocess.run(
            [str(CPU_GOLDEN_PATH), "matmul_mixed", "gfp16", "gfp8",
             str(a_path), str(b_path), str(c_path),
             str(M), str(K), str(N), "gfp16"],
            capture_output=True, text=True
        )
        assert result.returncode == 0, f"Failed: {result.stderr}"

        c_gfp = np.fromfile(c_path, dtype=np.uint16)
        c = gfloat16_to_fp32(c_gfp).reshape(M, N)

        assert c.shape == (M, N)


class TestMixedPrecisionCpuGolden:
    """混合精度 cpu_golden 测试"""

    def setup_method(self):
        skip_if_not_available()

    def test_matmul_mixed_cpu_golden(self):
        """MatMul 混合精度 cpu_golden"""
        from aidevtools.ops import _functional as F
        from aidevtools.ops.cpu_golden import set_cpu_golden_dtype

        # 设置混合精度: A 用 gfp8, B 用 gfp4, 输出用 gfp16
        set_cpu_golden_dtype(
            dtype="gfp16",
            dtype_matmul_a="gfp8",
            dtype_matmul_b="gfp4",
            dtype_matmul_out="gfp16"
        )

        M, K, N = 4, 8, 16
        a = np.random.randn(M, K).astype(np.float32)
        b = np.random.randn(K, N).astype(np.float32)

        op = F.MatMul()
        c = op.cpu_golden(a, b)

        assert c.shape == (M, N)
        assert c.dtype == np.float32

    def test_matmul_mixed_batch(self):
        """MatMul 混合精度 batch 支持"""
        from aidevtools.ops import _functional as F
        from aidevtools.ops.cpu_golden import set_cpu_golden_dtype

        set_cpu_golden_dtype(
            dtype="gfp16",
            dtype_matmul_a="gfp16",
            dtype_matmul_b="gfp8",
            dtype_matmul_out="gfp16"
        )

        # 3D batch matmul
        batch, M, K, N = 2, 4, 8, 16
        a = np.random.randn(batch, M, K).astype(np.float32)
        b = np.random.randn(K, N).astype(np.float32)  # broadcast

        op = F.MatMul()
        c = op.cpu_golden(a, b)

        assert c.shape == (batch, M, N)
        assert c.dtype == np.float32


class TestMixedPrecisionCppMode:
    """混合精度 cpp mode 测试"""

    def setup_method(self):
        skip_if_not_available()
        from aidevtools.ops.base import clear, set_golden_mode
        clear()
        set_golden_mode("python")

    def teardown_method(self):
        from aidevtools.ops.base import set_golden_mode
        from aidevtools.ops.cpu_golden import set_cpu_golden_dtype
        set_golden_mode("python")
        set_cpu_golden_dtype("gfp16")

    def test_mixed_precision_cpp_mode(self):
        """混合精度 cpp golden mode"""
        from aidevtools.ops import _functional as F
        from aidevtools.ops.cpu_golden import set_cpu_golden_dtype
        from aidevtools.ops.base import set_golden_mode, clear, get_records

        # 设置混合精度: A 用 gfp8, B 用 gfp4, 输出用 gfp16
        set_cpu_golden_dtype(
            dtype="gfp16",
            dtype_matmul_a="gfp8",
            dtype_matmul_b="gfp4",
            dtype_matmul_out="gfp16"
        )
        set_golden_mode("cpp")
        clear()

        # 执行 matmul
        a = np.random.randn(4, 8).astype(np.float32)
        b = np.random.randn(8, 16).astype(np.float32)
        c = F.matmul(a, b)

        assert c.shape == (4, 16)

        records = get_records()
        assert len(records) == 1
        assert records[0]["golden"] is not None


# ==================== TracedTensor 支持测试 ====================

class TestTracedTensorSupport:
    """TracedTensor 输入支持测试"""

    def setup_method(self):
        skip_if_not_available()
        from aidevtools.ops.cpu_golden import set_cpu_golden_dtype
        set_cpu_golden_dtype("gfp16")

    def test_unary_op_with_quantized_tensor(self):
        """单目运算支持 TracedTensor 输入"""
        from aidevtools.ops import _functional as F
        from aidevtools.ops.traced_tensor import TracedTensor

        x = np.random.randn(4, 8).astype(np.float32)

        # 创建已量化的 TracedTensor
        qt = TracedTensor(data=x, dtype="gfp16")

        # 使用 TracedTensor 调用 relu
        y = F.ReLU().cpu_golden(qt)

        assert y.shape == (4, 8)
        assert y.dtype == np.float32

    def test_unary_op_with_ndarray(self):
        """单目运算支持普通 ndarray 输入"""
        from aidevtools.ops import _functional as F

        x = np.random.randn(4, 8).astype(np.float32)

        y = F.GELU().cpu_golden(x)

        assert y.shape == (4, 8)
        assert y.dtype == np.float32

    def test_binary_op_with_quantized_tensor(self):
        """双目运算支持 TracedTensor 输入"""
        from aidevtools.ops import _functional as F
        from aidevtools.ops.traced_tensor import TracedTensor

        a = np.random.randn(4, 8).astype(np.float32)
        b = np.random.randn(4, 8).astype(np.float32)

        # 两个都是 TracedTensor
        qt_a = TracedTensor(data=a, dtype="gfp16")
        qt_b = TracedTensor(data=b, dtype="gfp16")

        c = F.Add().cpu_golden(qt_a, qt_b)

        assert c.shape == (4, 8)
        assert c.dtype == np.float32

    def test_binary_op_mixed_input(self):
        """双目运算支持混合输入 (TracedTensor + ndarray)"""
        from aidevtools.ops import _functional as F
        from aidevtools.ops.traced_tensor import TracedTensor

        a = np.random.randn(4, 8).astype(np.float32)
        b = np.random.randn(4, 8).astype(np.float32)

        # a 是 TracedTensor，b 是普通数组
        qt_a = TracedTensor(data=a, dtype="gfp16")

        c = F.Mul().cpu_golden(qt_a, b)

        assert c.shape == (4, 8)
        assert c.dtype == np.float32

    def test_quantized_tensor_dtype_mismatch(self):
        """TracedTensor 精度不匹配时正常工作"""
        from aidevtools.ops import _functional as F
        from aidevtools.ops.traced_tensor import TracedTensor
        from aidevtools.ops.cpu_golden import set_cpu_golden_dtype

        # 全局设置 gfp16
        set_cpu_golden_dtype("gfp16")

        x = np.random.randn(4, 8).astype(np.float32)

        # 创建 gfp8 的 TracedTensor，但全局是 gfp16
        qt = TracedTensor(data=x, dtype="gfp8")

        # 应该正常工作（会进行转换）
        y = F.Sigmoid().cpu_golden(qt)

        assert y.shape == (4, 8)
        assert y.dtype == np.float32

    def test_quantized_tensor_preserves_precision(self):
        """验证 TracedTensor 跳过重复转换

        当 TracedTensor.dtype 与全局 dtype 匹配时，
        应该直接使用内部数据，不进行额外转换。
        """
        from aidevtools.ops._functional import _extract_data
        from aidevtools.ops.traced_tensor import TracedTensor

        x = np.array([1.5, 2.5, 3.5], dtype=np.float32)

        # 创建匹配精度的 TracedTensor
        qt = TracedTensor(data=x, dtype="gfp16")

        # _extract_data 应该返回同一个数组对象（跳过转换）
        extracted = _extract_data(qt, "gfp16")

        # 验证是同一个对象（未复制）
        assert extracted is qt.data

    def test_extract_data_with_ndarray(self):
        """_extract_data 对普通数组的处理"""
        from aidevtools.ops._functional import _extract_data

        x = np.array([1.5, 2.5, 3.5], dtype=np.float32)

        extracted = _extract_data(x, "gfp16")

        assert extracted.dtype == np.float32
        np.testing.assert_array_equal(extracted, x)
