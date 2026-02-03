"""导出失败用例"""
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np

from aidevtools.core.log import logger
from aidevtools.formats.base import save


@dataclass
class ExportConfig:
    """导出配置"""
    output_dir: str
    op_name: str
    qsnr_threshold: float = 20.0


def _export_single_case(g_flat: np.ndarray, block: Dict, path: Path,
                        op_name: str, itemsize: int, dtype) -> bool:
    """导出单个失败用例，返回是否成功导出"""
    offset = block["offset"]
    size = block["size"]
    elem_start = offset // itemsize
    elem_end = elem_start + size // itemsize

    g_slice = g_flat[elem_start:elem_end]
    case_name = f"case_0x{offset:04x}"

    # 导出 bin
    save(str(path / f"{case_name}.bin"), g_slice, fmt="raw")

    # 导出 json
    param = {
        "op_name": op_name,
        "offset": offset,
        "size": size,
        "elem_start": elem_start,
        "elem_end": elem_end,
        "dtype": str(dtype),
        "shape": list(g_slice.shape),
        "qsnr": block.get("qsnr", 0),
        "max_abs": block.get("max_abs", 0),
    }
    (path / f"{case_name}.json").write_text(json.dumps(param, indent=2))
    return True


def export_failed_cases(golden: np.ndarray, blocks: List[Dict],
                        config: ExportConfig) -> int:
    """
    导出精度过低的片段

    Args:
        golden: golden 数据
        blocks: 分块比对结果
        config: 导出配置

    Returns:
        导出的用例数量
    """
    path = Path(config.output_dir) / config.op_name / "failed_cases"
    path.mkdir(parents=True, exist_ok=True)

    g_flat = golden.flatten()
    dtype = golden.dtype

    exported = sum(
        1 for block in blocks
        if block.get("qsnr", float("inf")) < config.qsnr_threshold
        and _export_single_case(g_flat, block, path, config.op_name, dtype.itemsize, dtype)
    )

    logger.info(f"[{config.op_name}] 导出 {exported} 个失败用例到 {path}")
    return exported
