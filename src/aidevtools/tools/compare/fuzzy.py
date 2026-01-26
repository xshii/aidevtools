"""模糊比对支持"""

from pathlib import Path
from typing import Any, Callable, Dict

import numpy as np

from aidevtools.core.log import logger
from aidevtools.formats.quantize import quantize


class FuzzyCase:
    """模糊比对用例"""

    def __init__(self, name: str, output_dir: str = "./workspace"):
        self.name = name
        self.output_dir = Path(output_dir) / name
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.inputs: Dict[str, np.ndarray] = {}
        self.weights: Dict[str, np.ndarray] = {}
        self.golden: np.ndarray = None
        self.compute_fn: Callable = None

    def set_input(self, name: str, data: np.ndarray):
        """设置输入数据 (fp32)"""
        self.inputs[name] = data.astype(np.float32)

    def set_weight(self, name: str, data: np.ndarray):
        """设置权重数据 (fp32)"""
        self.weights[name] = data.astype(np.float32)

    def set_compute(self, fn: Callable):
        """
        设置计算函数

        示例:
            def my_conv(inputs, weights):
                x = inputs["x"]
                w = weights["weight"]
                return np.matmul(x, w)

            case.set_compute(my_conv)
        """
        self.compute_fn = fn

    def compute_golden(self) -> np.ndarray:
        """执行 fp32 标准计算得到 Golden"""
        if self.compute_fn is None:
            raise ValueError("请先设置计算函数: set_compute(fn)")

        logger.info(f"[{self.name}] 执行 fp32 Golden 计算...")
        self.golden = self.compute_fn(self.inputs, self.weights)
        return self.golden

    def export(self, qtype: str = "float16", **kwargs) -> Dict[str, str]:
        """
        导出量化后的数据

        Args:
            qtype: 量化类型 (float16/bfloat16/int8_symmetric/...)
            **kwargs: 量化参数

        Returns:
            导出文件路径字典
        """
        paths = {}

        # 导出 Golden (保持 fp32)
        golden_path = self.output_dir / "golden.bin"
        self.golden.astype(np.float32).tofile(golden_path)
        paths["golden"] = str(golden_path)
        logger.info(f"[{self.name}] 导出 Golden: {golden_path}")

        # 导出量化后的输入
        for name, data in self.inputs.items():
            q_data, meta = quantize(data, qtype, **kwargs)
            out_path = self.output_dir / f"input_{name}.bin"
            q_data.tofile(out_path)
            paths[f"input_{name}"] = str(out_path)

            # 保存 meta 信息
            if meta:
                meta_path = self.output_dir / f"input_{name}.meta"
                with open(meta_path, "w", encoding="utf-8") as f:
                    for k, v in meta.items():
                        f.write(f"{k}={v}\n")

            logger.info(f"[{self.name}] 导出 input_{name}: {out_path} ({qtype})")

        # 导出量化后的权重
        for name, data in self.weights.items():
            q_data, meta = quantize(data, qtype, **kwargs)
            out_path = self.output_dir / f"weight_{name}.bin"
            q_data.tofile(out_path)
            paths[f"weight_{name}"] = str(out_path)

            if meta:
                meta_path = self.output_dir / f"weight_{name}.meta"
                with open(meta_path, "w", encoding="utf-8") as f:
                    for k, v in meta.items():
                        f.write(f"{k}={v}\n")

            logger.info(f"[{self.name}] 导出 weight_{name}: {out_path} ({qtype})")

        return paths

    def export_info(self) -> Dict[str, Any]:
        """导出用例信息"""
        return {
            "name": self.name,
            "inputs": {
                k: {"shape": list(v.shape), "dtype": str(v.dtype)} for k, v in self.inputs.items()
            },
            "weights": {
                k: {"shape": list(v.shape), "dtype": str(v.dtype)} for k, v in self.weights.items()
            },
            "golden_shape": list(self.golden.shape) if self.golden is not None else None,
        }


def create_fuzzy_case(name: str, output_dir: str = "./workspace") -> FuzzyCase:
    """创建模糊比对用例"""
    return FuzzyCase(name, output_dir)
