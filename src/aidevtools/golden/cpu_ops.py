"""CPU Golden Ops Python Wrapper

通过 subprocess 调用 cpu_golden 可执行文件，并封装为 Python 接口。
可直接用于 @register_golden_cpp 注册。

用法:
    from aidevtools.golden.cpu_ops import matmul, softmax, layernorm
    from aidevtools.ops.base import register_golden_cpp

    # 方式1: 直接注册
    register_golden_cpp("matmul")(matmul)
    register_golden_cpp("softmax")(softmax)
    register_golden_cpp("layernorm")(layernorm)

    # 方式2: 批量注册
    from aidevtools.golden.cpu_ops import register_all_cpu_golden
    register_all_cpu_golden()
"""
import subprocess
import tempfile
import numpy as np
from pathlib import Path
from typing import Optional, Literal

# CPU Golden 可执行文件路径
_CPU_GOLDEN_PATH = Path(__file__).parent / "cpu_golden"

GFloatType = Literal["gfp4", "gfp8", "gfp16"]


def _check_cpu_golden():
    """检查 cpu_golden 是否存在"""
    if not _CPU_GOLDEN_PATH.exists():
        raise FileNotFoundError(
            f"cpu_golden not found: {_CPU_GOLDEN_PATH}\n"
            f"Please build it first: cd {_CPU_GOLDEN_PATH.parent}/cpp && ./build.sh"
        )


def _fp32_to_gfloat(x: np.ndarray, dtype: GFloatType) -> np.ndarray:
    """fp32 转换为 gfloat 格式"""
    bits = x.astype(np.float32).view(np.uint32)
    if dtype == "gfp16":
        return (bits >> 16).astype(np.uint16)
    elif dtype == "gfp8":
        return (bits >> 24).astype(np.uint8)
    elif dtype == "gfp4":
        val4 = (bits >> 28).astype(np.uint8)
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
    else:
        raise ValueError(f"Unknown dtype: {dtype}")


def _gfloat_to_fp32(data: np.ndarray, dtype: GFloatType, size: Optional[int] = None) -> np.ndarray:
    """gfloat 格式转换为 fp32"""
    if dtype == "gfp16":
        bits = data.astype(np.uint32) << 16
        return bits.view(np.float32)
    elif dtype == "gfp8":
        bits = data.astype(np.uint32) << 24
        return bits.view(np.float32)
    elif dtype == "gfp4":
        if size is None:
            size = data.size * 2
        output = np.zeros(size, dtype=np.float32)
        for i in range(size):
            byte_idx = i // 2
            if i % 2 == 0:
                val4 = (data[byte_idx] >> 4) & 0x0F
            else:
                val4 = data[byte_idx] & 0x0F
            bits = np.uint32(val4) << 28
            output[i] = np.array([bits], dtype=np.uint32).view(np.float32)[0]
        return output
    else:
        raise ValueError(f"Unknown dtype: {dtype}")


def _get_gfloat_numpy_dtype(dtype: GFloatType):
    """获取 gfloat 对应的 numpy dtype"""
    if dtype == "gfp16":
        return np.uint16
    else:
        return np.uint8


def matmul(
    a: np.ndarray,
    b: np.ndarray,
    dtype: GFloatType = "gfp16",
    dtype_a: Optional[GFloatType] = None,
    dtype_b: Optional[GFloatType] = None,
    dtype_out: Optional[GFloatType] = None,
) -> np.ndarray:
    """
    MatMul: C = A @ B (支持 batch 和混合精度)

    Args:
        a: 输入矩阵 A, shape [..., M, K]
        b: 输入矩阵 B, shape [..., K, N] 或 [K, N]
        dtype: gfloat 类型 (gfp4, gfp8, gfp16)，当 dtype_a/dtype_b 未指定时使用
        dtype_a: A 矩阵的 gfloat 类型 (混合精度)
        dtype_b: B 矩阵的 gfloat 类型 (混合精度)
        dtype_out: 输出矩阵的 gfloat 类型 (默认与 dtype 相同)

    Returns:
        输出矩阵 C, shape [..., M, N]

    Example:
        # 同精度
        c = matmul(a, b, dtype="gfp16")

        # 混合精度: A 用 gfp8, B 用 gfp4, 输出用 gfp16
        c = matmul(a, b, dtype_a="gfp8", dtype_b="gfp4", dtype_out="gfp16")
    """
    _check_cpu_golden()

    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)

    # 确定 dtype_a, dtype_b, dtype_out
    if dtype_a is None:
        dtype_a = dtype
    if dtype_b is None:
        dtype_b = dtype
    if dtype_out is None:
        dtype_out = dtype

    # 是否混合精度
    is_mixed = (dtype_a != dtype_b) or (dtype_a != dtype_out)

    # 处理 batch 维度
    a_batch_shape = a.shape[:-2] if a.ndim > 2 else ()
    b_batch_shape = b.shape[:-2] if b.ndim > 2 else ()

    # 获取 M, K, N
    M, K = a.shape[-2:]
    if b.ndim == 1:
        K2, N = b.shape[0], 1
        b = b.reshape(K2, N)
    else:
        K2, N = b.shape[-2:]

    assert K == K2, f"Shape mismatch: a.shape={a.shape}, b.shape={b.shape}"

    # 处理 2D 情况 (快速路径)
    if a.ndim == 2 and b.ndim == 2:
        if is_mixed:
            return _matmul_2d_mixed(a, b, M, K, N, dtype_a, dtype_b, dtype_out)
        else:
            return _matmul_2d(a, b, M, K, N, dtype)

    # 处理 batch: flatten batch dims, 循环调用 2D matmul
    if a.ndim > 2:
        batch_size = int(np.prod(a_batch_shape))
        a_flat = a.reshape(batch_size, M, K)
    else:
        batch_size = 1
        a_flat = a.reshape(1, M, K)

    # b 可能没有 batch (广播)
    if b.ndim == 2:
        b_batched = False
        b_flat = b.reshape(1, K, N)
    else:
        b_batched = True
        b_flat = b.reshape(batch_size, K, N)

    # 逐 batch 计算
    c_flat = np.zeros((batch_size, M, N), dtype=np.float32)
    for i in range(batch_size):
        a_i = a_flat[i]
        b_i = b_flat[i] if b_batched else b_flat[0]
        if is_mixed:
            c_flat[i] = _matmul_2d_mixed(a_i, b_i, M, K, N, dtype_a, dtype_b, dtype_out)
        else:
            c_flat[i] = _matmul_2d(a_i, b_i, M, K, N, dtype)

    # 恢复 batch shape
    output_shape = a_batch_shape + (M, N)
    return c_flat.reshape(output_shape)


def _matmul_2d(a: np.ndarray, b: np.ndarray, M: int, K: int, N: int, dtype: GFloatType) -> np.ndarray:
    """2D 矩阵乘法 (内部函数)"""
    with tempfile.TemporaryDirectory() as tmpdir:
        a_path = Path(tmpdir) / "a.bin"
        b_path = Path(tmpdir) / "b.bin"
        c_path = Path(tmpdir) / "c.bin"

        # 保存输入
        _fp32_to_gfloat(a, dtype).tofile(a_path)
        _fp32_to_gfloat(b, dtype).tofile(b_path)

        # 调用 cpu_golden
        result = subprocess.run(
            [str(_CPU_GOLDEN_PATH), "matmul", dtype,
             str(a_path), str(b_path), str(c_path),
             str(M), str(K), str(N)],
            capture_output=True, text=True
        )

        if result.returncode != 0:
            raise RuntimeError(f"cpu_golden matmul failed: {result.stderr}")

        # 读取结果
        np_dtype = _get_gfloat_numpy_dtype(dtype)
        c_gfp = np.fromfile(c_path, dtype=np_dtype)
        c = _gfloat_to_fp32(c_gfp, dtype, M * N).reshape(M, N)

    return c


def _matmul_2d_mixed(
    a: np.ndarray, b: np.ndarray, M: int, K: int, N: int,
    dtype_a: GFloatType, dtype_b: GFloatType, dtype_out: GFloatType
) -> np.ndarray:
    """2D 混合精度矩阵乘法 (内部函数)"""
    with tempfile.TemporaryDirectory() as tmpdir:
        a_path = Path(tmpdir) / "a.bin"
        b_path = Path(tmpdir) / "b.bin"
        c_path = Path(tmpdir) / "c.bin"

        # 保存输入 (各自使用自己的 dtype)
        _fp32_to_gfloat(a, dtype_a).tofile(a_path)
        _fp32_to_gfloat(b, dtype_b).tofile(b_path)

        # 调用 cpu_golden matmul_mixed
        result = subprocess.run(
            [str(_CPU_GOLDEN_PATH), "matmul_mixed",
             dtype_a, dtype_b,
             str(a_path), str(b_path), str(c_path),
             str(M), str(K), str(N), dtype_out],
            capture_output=True, text=True
        )

        if result.returncode != 0:
            raise RuntimeError(f"cpu_golden matmul_mixed failed: {result.stderr}")

        # 读取结果 (使用 dtype_out)
        np_dtype = _get_gfloat_numpy_dtype(dtype_out)
        c_gfp = np.fromfile(c_path, dtype=np_dtype)
        c = _gfloat_to_fp32(c_gfp, dtype_out, M * N).reshape(M, N)

    return c


def softmax(x: np.ndarray, dtype: GFloatType = "gfp16") -> np.ndarray:
    """
    Softmax: y = softmax(x, axis=-1)

    Args:
        x: 输入数组, shape [..., seq]
        dtype: gfloat 类型 (gfp4, gfp8, gfp16)

    Returns:
        输出数组, shape [..., seq]
    """
    _check_cpu_golden()

    x = np.asarray(x, dtype=np.float32)
    original_shape = x.shape

    # flatten 到 2D: [batch, seq]
    if x.ndim == 1:
        x = x.reshape(1, -1)
    elif x.ndim > 2:
        x = x.reshape(-1, x.shape[-1])

    batch, seq = x.shape

    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = Path(tmpdir) / "input.bin"
        output_path = Path(tmpdir) / "output.bin"

        _fp32_to_gfloat(x, dtype).tofile(input_path)

        result = subprocess.run(
            [str(_CPU_GOLDEN_PATH), "softmax", dtype,
             str(input_path), str(output_path),
             str(batch), str(seq)],
            capture_output=True, text=True
        )

        if result.returncode != 0:
            raise RuntimeError(f"cpu_golden softmax failed: {result.stderr}")

        np_dtype = _get_gfloat_numpy_dtype(dtype)
        y_gfp = np.fromfile(output_path, dtype=np_dtype)
        y = _gfloat_to_fp32(y_gfp, dtype, batch * seq).reshape(batch, seq)

    return y.reshape(original_shape)


def layernorm(
    x: np.ndarray,
    gamma: np.ndarray,
    beta: np.ndarray,
    dtype: GFloatType = "gfp16",
    eps: float = 1e-5
) -> np.ndarray:
    """
    LayerNorm: y = (x - mean) / sqrt(var + eps) * gamma + beta

    Args:
        x: 输入数组, shape [..., hidden]
        gamma: 缩放参数, shape [hidden]
        beta: 偏移参数, shape [hidden]
        dtype: gfloat 类型 (gfp4, gfp8, gfp16)
        eps: 数值稳定性参数 (注意: cpu_golden 使用固定 eps=1e-5)

    Returns:
        输出数组, shape [..., hidden]
    """
    _check_cpu_golden()

    x = np.asarray(x, dtype=np.float32)
    gamma = np.asarray(gamma, dtype=np.float32)
    beta = np.asarray(beta, dtype=np.float32)

    original_shape = x.shape
    hidden = x.shape[-1]

    # flatten 到 2D: [batch, hidden]
    if x.ndim == 1:
        x = x.reshape(1, -1)
    elif x.ndim > 2:
        x = x.reshape(-1, hidden)

    batch = x.shape[0]

    assert gamma.shape == (hidden,), f"gamma shape mismatch: {gamma.shape} vs ({hidden},)"
    assert beta.shape == (hidden,), f"beta shape mismatch: {beta.shape} vs ({hidden},)"

    with tempfile.TemporaryDirectory() as tmpdir:
        x_path = Path(tmpdir) / "x.bin"
        gamma_path = Path(tmpdir) / "gamma.bin"
        beta_path = Path(tmpdir) / "beta.bin"
        y_path = Path(tmpdir) / "y.bin"

        _fp32_to_gfloat(x, dtype).tofile(x_path)
        _fp32_to_gfloat(gamma, dtype).tofile(gamma_path)
        _fp32_to_gfloat(beta, dtype).tofile(beta_path)

        result = subprocess.run(
            [str(_CPU_GOLDEN_PATH), "layernorm", dtype,
             str(x_path), str(gamma_path), str(beta_path), str(y_path),
             str(batch), str(hidden)],
            capture_output=True, text=True
        )

        if result.returncode != 0:
            raise RuntimeError(f"cpu_golden layernorm failed: {result.stderr}")

        np_dtype = _get_gfloat_numpy_dtype(dtype)
        y_gfp = np.fromfile(y_path, dtype=np_dtype)
        y = _gfloat_to_fp32(y_gfp, dtype, batch * hidden).reshape(batch, hidden)

    return y.reshape(original_shape)


def transpose(
    x: np.ndarray,
    dtype: GFloatType = "gfp16",
) -> np.ndarray:
    """
    Transpose 4D: 交换最后两个维度

    Args:
        x: 输入数组, shape [d0, d1, d2, d3]
        dtype: gfloat 类型 (gfp4, gfp8, gfp16)

    Returns:
        输出数组, shape [d0, d1, d3, d2]
    """
    _check_cpu_golden()

    x = np.asarray(x, dtype=np.float32)

    if x.ndim != 4:
        raise ValueError(f"transpose requires 4D input, got {x.ndim}D")

    d0, d1, d2, d3 = x.shape

    with tempfile.TemporaryDirectory() as tmpdir:
        x_path = Path(tmpdir) / "x.bin"
        y_path = Path(tmpdir) / "y.bin"

        _fp32_to_gfloat(x, dtype).tofile(x_path)

        result = subprocess.run(
            [str(_CPU_GOLDEN_PATH), "transpose", dtype,
             str(x_path), str(y_path),
             str(d0), str(d1), str(d2), str(d3)],
            capture_output=True, text=True
        )

        if result.returncode != 0:
            raise RuntimeError(f"cpu_golden transpose failed: {result.stderr}")

        np_dtype = _get_gfloat_numpy_dtype(dtype)
        y_gfp = np.fromfile(y_path, dtype=np_dtype)
        y = _gfloat_to_fp32(y_gfp, dtype, d0 * d1 * d2 * d3)

    # 输出 shape: [d0, d1, d3, d2]
    return y.reshape(d0, d1, d3, d2)


def register_all_cpu_golden(
    dtype: GFloatType = "gfp16",
    dtype_matmul_a: Optional[GFloatType] = None,
    dtype_matmul_b: Optional[GFloatType] = None,
    dtype_matmul_out: Optional[GFloatType] = None,
):
    """
    批量注册所有 CPU Golden 算子

    Args:
        dtype: 默认 gfloat 类型
        dtype_matmul_a: matmul 的 A 矩阵类型 (混合精度)
        dtype_matmul_b: matmul 的 B 矩阵类型 (混合精度)
        dtype_matmul_out: matmul 的输出类型 (混合精度)

    用法:
        from aidevtools.golden.cpu_ops import register_all_cpu_golden
        register_all_cpu_golden("gfp16")

        # 混合精度 matmul: A 用 gfp8, B 用 gfp4, 输出用 gfp16
        register_all_cpu_golden(
            dtype="gfp16",
            dtype_matmul_a="gfp8",
            dtype_matmul_b="gfp4",
            dtype_matmul_out="gfp16"
        )

        # 之后设置 golden_mode="cpp" 即可使用
        from aidevtools.ops import set_golden_mode
        set_golden_mode("cpp")
    """
    from aidevtools.ops.base import register_golden_cpp

    # matmul 的 dtype 配置
    ma = dtype_matmul_a or dtype
    mb = dtype_matmul_b or dtype
    mo = dtype_matmul_out or dtype

    # 创建带默认 dtype 的闭包
    @register_golden_cpp("matmul")
    def _matmul(a, b):
        return matmul(a, b, dtype=dtype, dtype_a=ma, dtype_b=mb, dtype_out=mo)

    @register_golden_cpp("softmax")
    def _softmax(x, axis=-1):
        # cpp softmax 沿最后一维计算，忽略 axis 参数
        return softmax(x, dtype=dtype)

    @register_golden_cpp("layernorm")
    def _layernorm(x, gamma, beta, eps=1e-5):
        return layernorm(x, gamma, beta, dtype=dtype, eps=eps)

    @register_golden_cpp("transpose")
    def _transpose(x, axes=None):
        # CPU golden 只支持 4D 交换最后两维
        return transpose(x, dtype=dtype)


def is_cpu_golden_available() -> bool:
    """检查 cpu_golden 是否可用"""
    return _CPU_GOLDEN_PATH.exists()
