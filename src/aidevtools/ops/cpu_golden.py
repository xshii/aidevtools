"""CPU Golden 基础设施

提供 C++ Golden 调用所需的基础设施：
- gfloat 格式转换
- subprocess 通用执行函数
- 全局配置（dtype 等）

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
"""
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple

import numpy as np

# CPU Golden 可执行文件路径 (在 golden 目录)
_GOLDEN_DIR = Path(__file__).parent.parent / "golden"
_CPU_GOLDEN_PATH = _GOLDEN_DIR / "cpu_golden"
_CPP_DIR = _GOLDEN_DIR / "cpp"

from aidevtools.core.config import get_config, set_config

GFloatType = Literal["gfp4", "gfp8", "gfp16"]


# ============================================================
# 全局配置
# ============================================================

def set_cpu_golden_dtype(
    dtype: GFloatType = "gfp16",
    dtype_matmul_a: Optional[GFloatType] = None,
    dtype_matmul_b: Optional[GFloatType] = None,
    dtype_matmul_out: Optional[GFloatType] = None,
):
    """
    设置 CPU Golden 全局 dtype 配置

    Args:
        dtype: 默认 gfloat 类型
        dtype_matmul_a: matmul 的 A 矩阵类型 (混合精度)
        dtype_matmul_b: matmul 的 B 矩阵类型 (混合精度)
        dtype_matmul_out: matmul 的输出类型 (混合精度)

    用法:
        # 同精度
        set_cpu_golden_dtype("gfp16")

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
    """检查 cpu_golden 是否可用"""
    return _CPU_GOLDEN_PATH.exists()


def _check_cpu_golden():
    """检查 cpu_golden 是否存在"""
    import os
    import stat

    if not _CPU_GOLDEN_PATH.exists():
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
            f"CPU Golden 可执行文件未找到\n"
            f"{'=' * 50}\n"
            f"原因: {detail}\n"
            f"期望路径: {_CPU_GOLDEN_PATH}\n"
            f"{'=' * 50}\n"
            f"解决方法:\n"
            f"  cd {_CPP_DIR}\n"
            f"  ./build.sh\n"
        )

    # 检查是否可执行
    if not os.access(_CPU_GOLDEN_PATH, os.X_OK):
        file_stat = os.stat(_CPU_GOLDEN_PATH)
        mode = stat.filemode(file_stat.st_mode)
        raise PermissionError(
            f"CPU Golden 文件存在但没有执行权限\n"
            f"{'=' * 50}\n"
            f"文件: {_CPU_GOLDEN_PATH}\n"
            f"权限: {mode}\n"
            f"{'=' * 50}\n"
            f"解决方法:\n"
            f"  chmod +x {_CPU_GOLDEN_PATH}\n"
        )


# ============================================================
# gfloat 格式转换
# ============================================================

def _fp32_to_gfloat(x: np.ndarray, dtype: GFloatType) -> np.ndarray:
    """fp32 转换为 gfloat 格式"""
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


def _gfloat_to_fp32(data: np.ndarray, dtype: GFloatType, size: Optional[int] = None) -> np.ndarray:
    """gfloat 格式转换为 fp32"""
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


def _get_gfloat_numpy_dtype(dtype: GFloatType):
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
    inputs: Dict[str, Tuple[np.ndarray, GFloatType]],
    output_name: str,
    output_dtype: GFloatType,
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
        output_dtype: 输出数据的 gfloat 类型
        output_size: 输出元素数量
        output_shape: 输出 shape

    Returns:
        输出数组
    """
    _check_cpu_golden()

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # 保存输入文件
        input_paths = {}
        for name, (arr, dtype) in inputs.items():
            path = tmpdir / name
            _fp32_to_gfloat(arr, dtype).tofile(path)
            input_paths[name] = str(path)

        # 输出路径
        output_path = tmpdir / output_name

        # 构建完整命令 (替换占位符)
        full_cmd = [str(_CPU_GOLDEN_PATH)]
        for arg in cmd_args:
            if arg == "@output":
                # @output -> 输出文件路径
                full_cmd.append(str(output_path))
            elif arg.startswith("@"):
                # @input_name -> 对应输入文件的路径
                full_cmd.append(input_paths.get(arg[1:], str(tmpdir / arg[1:])))
            else:
                full_cmd.append(arg)

        # 执行 subprocess
        result = subprocess.run(full_cmd, capture_output=True, text=True)

        if result.returncode != 0:
            raise RuntimeError(f"cpu_golden {op_name} failed: {result.stderr}")

        # 读取输出
        np_dtype = _get_gfloat_numpy_dtype(output_dtype)
        out_gfp = np.fromfile(output_path, dtype=np_dtype)
        out = _gfloat_to_fp32(out_gfp, output_dtype, output_size)

    return out.reshape(output_shape)
