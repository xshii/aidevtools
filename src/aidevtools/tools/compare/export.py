"""导出失败用例"""
import json
from pathlib import Path
from typing import List, Dict
import numpy as np

from aidevtools.core.log import logger
from aidevtools.formats.base import save

def export_failed_cases(golden: np.ndarray, result: np.ndarray,
                        blocks: List[Dict], output_dir: str,
                        op_name: str, qsnr_threshold: float = 20.0):
    """
    导出精度过低的片段

    Args:
        golden: golden 数据
        result: 待比对数据
        blocks: 分块比对结果
        output_dir: 输出目录
        op_name: 算子名
        qsnr_threshold: QSNR 阈值 (低于此值导出)
    """
    path = Path(output_dir) / op_name / "failed_cases"
    path.mkdir(parents=True, exist_ok=True)

    g_flat = golden.flatten()
    r_flat = result.flatten()
    dtype = golden.dtype
    itemsize = dtype.itemsize

    exported = 0
    for block in blocks:
        qsnr = block.get("qsnr", float("inf"))
        if qsnr >= qsnr_threshold:
            continue

        offset = block["offset"]
        size = block["size"]

        # 计算元素范围
        elem_start = offset // itemsize
        elem_end = elem_start + size // itemsize

        # 导出 golden 片段
        g_slice = g_flat[elem_start:elem_end]
        case_name = f"case_0x{offset:04x}"
        bin_path = path / f"{case_name}.bin"
        save(str(bin_path), g_slice, format="raw")

        # 导出参数
        param = {
            "op_name": op_name,
            "offset": offset,
            "size": size,
            "elem_start": elem_start,
            "elem_end": elem_end,
            "dtype": str(dtype),
            "shape": list(g_slice.shape),
            "qsnr": qsnr,
            "max_abs": block.get("max_abs", 0),
        }
        json_path = path / f"{case_name}.json"
        json_path.write_text(json.dumps(param, indent=2))

        exported += 1

    logger.info(f"[{op_name}] 导出 {exported} 个失败用例到 {path}")
    return exported
