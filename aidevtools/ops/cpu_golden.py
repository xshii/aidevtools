"""CPU Golden 基础设施

提供 C++ Golden 调用所需的基础设施：
- gfloat 格式转换
- subprocess 通用执行函数
- 全局配置（dtype 等）
- TracedTensor 支持

用法:
    from aidevtools.ops.cpu_golden import (
        run_cpu_golden,
        set_cpu_golden_dtype,
        get_cpu_golden_dtype,
    )

    # 设置全局 dtype
    set_cpu_golden_dtype("gfp16")

    # 在算子类中调用
    result = run_cpu_golden(
        op_name="matmul",
        cmd_args=["matmul", dtype, "@a.bin", "@b.bin", "@output", str(M), str(K), str(N)],
        inputs={"a.bin": (a, dtype), "b.bin": (b, dtype)},
        output_name="c.bin",
        output_dtype=dtype,
        output_size=M * N,
        output_shape=(M, N),
    )

使用 TracedTensor:
    from aidevtools.ops import quantize, TracedTensor

    # 在数据源头量化一次
    x = quantize(input_data, "gfp16")
    w = quantize(weight_data, "gfp16")

    # 后续计算自动保持精度状态
    y = F.matmul(x.numpy(), w.numpy())
"""
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple

import numpy as np

from aidevtools.core.config import get_config, set_config

# CPU Golden 可执行文件路径 (在 golden 目录)
_GOLDEN_DIR = Path(__file__).parent.parent / "golden"
_CPU_GOLDEN_PATH = _GOLDEN_DIR / "cpu_golden"
_CPU_GOLDEN_BFP_PATH = _GOLDEN_DIR / "cpu_golden_bfp"
_CPP_DIR = _GOLDEN_DIR / "cpp"

GFloatType = Literal["gfp4", "gfp8", "gfp16"]
BFPType = Literal["bfp4", "bfp8", "bfp16"]
QuantizedType = Literal["gfp4", "gfp8", "gfp16", "bfp4", "bfp8", "bfp16"]


def _is_bfp_type(dtype: str) -> bool:
    """判断是否是 BFP 类型"""
    return dtype in ("bfp4", "bfp8", "bfp16")


def _get_executable(dtype: str) -> Path:
    """根据 dtype 获取对应的可执行文件"""
    if _is_bfp_type(dtype):
        return _CPU_GOLDEN_BFP_PATH
    return _CPU_GOLDEN_PATH


# ============================================================
# 全局配置
# ============================================================

def set_cpu_golden_dtype(
    dtype: QuantizedType = "gfp16",
    dtype_matmul_a: Optional[QuantizedType] = None,
    dtype_matmul_b: Optional[QuantizedType] = None,
    dtype_matmul_out: Optional[QuantizedType] = None,
):
    """
    设置 CPU Golden 全局 dtype 配置

    Args:
        dtype: 量化类型 (gfp4/gfp8/gfp16/bfp4/bfp8/bfp16)
        dtype_matmul_a: matmul 的 A 矩阵类型 (混合精度)
        dtype_matmul_b: matmul 的 B 矩阵类型 (混合精度)
        dtype_matmul_out: matmul 的输出类型 (混合精度)

    用法:
        # GFloat 精度
        set_cpu_golden_dtype("gfp16")

        # BFP 精度
        set_cpu_golden_dtype("bfp8")

        # 混合精度 matmul: A 用 gfp8, B 用 gfp4, 输出用 gfp16
        set_cpu_golden_dtype(
            dtype="gfp16",
            dtype_matmul_a="gfp8",
            dtype_matmul_b="gfp4",
            dtype_matmul_out="gfp16"
        )

    注意: 也可使用 set_config(cpu_golden=CpuGoldenConfig(...)) 统一设置
    """
    from aidevtools.core.config import CpuGoldenConfig
    set_config(cpu_golden=CpuGoldenConfig(
        dtype=dtype,
        dtype_matmul_a=dtype_matmul_a,
        dtype_matmul_b=dtype_matmul_b,
        dtype_matmul_out=dtype_matmul_out,
    ))


def get_cpu_golden_dtype() -> GFloatType:
    """获取当前 CPU Golden dtype"""
    return get_config().cpu_golden.dtype


def get_matmul_dtypes() -> Tuple[GFloatType, GFloatType, GFloatType]:
    """获取 matmul 混合精度配置"""
    cfg = get_config().cpu_golden
    dtype = cfg.dtype
    return (
        cfg.dtype_matmul_a or dtype,
        cfg.dtype_matmul_b or dtype,
        cfg.dtype_matmul_out or dtype,
    )


# ============================================================
# 检查与路径
# ============================================================

def is_cpu_golden_available() -> bool:
    """检查 cpu_golden (GFloat) 是否可用"""
    return _CPU_GOLDEN_PATH.exists()


def is_bfp_available() -> bool:
    """检查 cpu_golden_bfp (BFP) 是否可用"""
    return _CPU_GOLDEN_BFP_PATH.exists()


def _check_cpu_golden(dtype: str = "gfp16"):
    """检查 cpu_golden 是否存在

    Args:
        dtype: 精度类型，用于选择正确的可执行文件
    """
    import os
    import stat

    executable = _get_executable(dtype)
    exe_name = "cpu_golden_bfp" if _is_bfp_type(dtype) else "cpu_golden"

    if not executable.exists():
        # 检查目录是否存在
        if not _GOLDEN_DIR.exists():
            detail = f"目录不存在: {_GOLDEN_DIR}"
        elif not _CPP_DIR.exists():
            detail = f"源码目录不存在: {_CPP_DIR}"
        else:
            # 列出目录内容帮助诊断
            files = list(_GOLDEN_DIR.glob("*"))
            detail = f"目录存在但缺少可执行文件\n  目录内容: {[f.name for f in files]}"

        raise FileNotFoundError(
            f"CPU Golden 可执行文件未找到 ({exe_name})\n"
            f"{'=' * 50}\n"
            f"原因: {detail}\n"
            f"期望路径: {executable}\n"
            f"{'=' * 50}\n"
            f"解决方法:\n"
            f"  cd {_CPP_DIR}\n"
            f"  ./build.sh\n"
        )

    # 检查是否可执行
    if not os.access(executable, os.X_OK):
        file_stat = os.stat(executable)
        mode = stat.filemode(file_stat.st_mode)
        raise PermissionError(
            f"CPU Golden 文件存在但没有执行权限 ({exe_name})\n"
            f"{'=' * 50}\n"
            f"文件: {executable}\n"
            f"权限: {mode}\n"
            f"{'=' * 50}\n"
            f"解决方法:\n"
            f"  chmod +x {executable}\n"
        )


# ============================================================
# 格式转换 (GFloat 和 BFP)
# ============================================================

def _get_bfp_params(dtype: str) -> Tuple[int, int]:
    """获取 BFP 参数

    Returns:
        (block_size, mantissa_bits)
    """
    if dtype == "bfp4":
        return 64, 2
    if dtype == "bfp8":
        return 32, 4
    if dtype == "bfp16":
        return 16, 8
    raise ValueError(f"Unknown BFP dtype: {dtype}")


def _fp32_to_bfp(x: np.ndarray, dtype: str) -> np.ndarray:
    """fp32 转换为 BFP 格式（单文件格式）

    通过调用 C++ encode 命令实现：
        ./cpu_golden_bfp encode <dtype> <input_fp32.bin> <output_packed.bin> <size>

    文件格式: [shared_exps (num_blocks 个 int8)] [mantissas (size 个 int8)]

    Args:
        x: 输入 fp32 数组
        dtype: BFP 类型 (bfp4/bfp8/bfp16)

    Returns:
        打包后的 int8 数组（exp + mantissa）
    """
    _check_cpu_golden(dtype)
    executable = _get_executable(dtype)

    flat = x.astype(np.float32).flatten()
    size = flat.size

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        input_path = tmpdir / "input.bin"
        output_path = tmpdir / "output.bin"

        # 保存 fp32 输入
        flat.tofile(input_path)

        # 调用 C++ encode
        cmd = [str(executable), "encode", dtype, str(input_path), str(output_path), str(size)]
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)

        if result.returncode != 0:
            raise RuntimeError(f"BFP encode failed: {result.stderr}")

        # 读取打包后的 BFP 数据
        packed = np.fromfile(output_path, dtype=np.int8)

    return packed


def _bfp_to_fp32(packed: np.ndarray, dtype: str, size: int) -> np.ndarray:
    """BFP 格式转换为 fp32（单文件格式）

    通过调用 C++ decode 命令实现：
        ./cpu_golden_bfp decode <dtype> <input_packed.bin> <output_fp32.bin> <size>

    文件格式: [shared_exps (num_blocks 个 int8)] [mantissas (size 个 int8)]

    Args:
        packed: 打包的 int8 数组（exp + mantissa）
        dtype: BFP 类型
        size: 输出元素数量

    Returns:
        fp32 数组
    """
    _check_cpu_golden(dtype)
    executable = _get_executable(dtype)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        input_path = tmpdir / "input.bin"
        output_path = tmpdir / "output.bin"

        # 保存打包的 BFP 数据
        packed.tofile(input_path)

        # 调用 C++ decode
        cmd = [str(executable), "decode", dtype, str(input_path), str(output_path), str(size)]
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)

        if result.returncode != 0:
            raise RuntimeError(f"BFP decode failed: {result.stderr}")

        # 读取 fp32 输出
        output = np.fromfile(output_path, dtype=np.float32)

    return output


def _fp32_to_gfloat(x: np.ndarray, dtype: str) -> np.ndarray:
    """fp32 转换为 gfloat 格式

    注意: 此函数只处理 GFloat 格式，BFP 使用 _fp32_to_bfp
    """
    # GFloat 格式转换
    bits = x.astype(np.float32).view(np.uint32)
    if dtype == "gfp16":
        return (bits >> 16).astype(np.uint16)
    if dtype == "gfp8":
        return (bits >> 24).astype(np.uint8)
    if dtype == "gfp4":
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
    raise ValueError(f"Unknown dtype: {dtype}")


def _gfloat_to_fp32(data: np.ndarray, dtype: str, size: Optional[int] = None) -> np.ndarray:
    """gfloat 格式转换为 fp32

    注意: 此函数只处理 GFloat 格式，BFP 使用 _bfp_to_fp32
    """
    # GFloat 格式转换
    if dtype == "gfp16":
        bits = data.astype(np.uint32) << 16
        return bits.view(np.float32)
    if dtype == "gfp8":
        bits = data.astype(np.uint32) << 24
        return bits.view(np.float32)
    if dtype == "gfp4":
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
    raise ValueError(f"Unknown dtype: {dtype}")


def _get_gfloat_numpy_dtype(dtype: str):
    """获取 gfloat 对应的 numpy dtype"""
    if dtype == "gfp16":
        return np.uint16
    return np.uint8


# ============================================================
# 通用 subprocess 执行函数
# ============================================================

def run_cpu_golden(
    op_name: str,
    cmd_args: List[str],
    inputs: Dict[str, Tuple[np.ndarray, str]],
    output_name: str,
    output_dtype: str,
    output_size: int,
    output_shape: Tuple[int, ...],
) -> np.ndarray:
    """
    通用 CPU Golden 执行函数

    Args:
        op_name: 算子名称 (用于错误信息)
        cmd_args: 命令行参数 (不含输入输出文件路径)
        inputs: 输入数据 {文件名: (数组, dtype)}
        output_name: 输出文件名
        output_dtype: 输出数据的精度类型 (gfp4/gfp8/gfp16/bfp4/bfp8/bfp16)
        output_size: 输出元素数量
        output_shape: 输出 shape

    Returns:
        输出数组
    """
    # 判断是用 GFloat 还是 BFP 可执行文件
    use_bfp = _is_bfp_type(output_dtype)
    primary_dtype = output_dtype

    _check_cpu_golden(primary_dtype)
    executable = _get_executable(primary_dtype)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # 保存输入文件
        input_paths = {}
        for name, (arr, dtype) in inputs.items():
            path = tmpdir / name
            if _is_bfp_type(dtype):
                # BFP: 单文件格式 [exps][mantissas]
                _fp32_to_bfp(arr, dtype).tofile(path)
            else:
                # GFloat: 压缩格式
                _fp32_to_gfloat(arr, dtype).tofile(path)
            input_paths[name] = str(path)

        # 输出路径
        output_path = tmpdir / output_name

        # 构建完整命令 (替换占位符)
        full_cmd = [str(executable)]
        for arg in cmd_args:
            if arg == "@output":
                full_cmd.append(str(output_path))
            elif arg.startswith("@"):
                full_cmd.append(input_paths.get(arg[1:], str(tmpdir / arg[1:])))
            else:
                full_cmd.append(arg)

        # 执行 subprocess
        result = subprocess.run(full_cmd, capture_output=True, text=True, check=False)

        if result.returncode != 0:
            raise RuntimeError(f"cpu_golden {op_name} failed: {result.stderr}")

        # 读取输出
        if use_bfp:
            # BFP: 读取单文件格式 [exps][mantissas]
            packed = np.fromfile(output_path, dtype=np.int8)
            out = _bfp_to_fp32(packed, output_dtype, output_size)
        else:
            # GFloat: 读取压缩格式
            np_dtype = _get_gfloat_numpy_dtype(output_dtype)
            out_gfp = np.fromfile(output_path, dtype=np_dtype)
            out = _gfloat_to_fp32(out_gfp, output_dtype, output_size)

    return out.reshape(output_shape)
