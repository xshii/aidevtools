"""执行引擎"""
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np

from aidevtools.core.config import get_config
from aidevtools.core.tensor import Tensor, generate_random, generate_weight
from aidevtools.core.op import run_golden, get_op
from aidevtools.core.log import logger


@dataclass
class OpRecord:
    """算子执行记录"""
    id: int
    op_name: str
    qtype: str

    # 输入输出 (Tensor 格式)
    inputs: List[Tensor] = field(default_factory=list)
    weights: List[Tensor] = field(default_factory=list)
    output: Optional[Tensor] = None

    # Golden 输出
    golden_pure: Optional[np.ndarray] = None    # 纯 fp32 计算结果
    golden_quant: Optional[np.ndarray] = None   # 量化感知计算结果

    # 元信息
    seed: Optional[int] = None
    note: str = ""


class ExecutionEngine:
    """执行引擎"""

    def __init__(self):
        self.records: List[OpRecord] = []
        self._counter: Dict[str, int] = {}

    def clear(self):
        """清空记录"""
        self.records.clear()
        self._counter.clear()

    def _get_op_id(self, op_name: str) -> int:
        """获取算子 ID"""
        idx = self._counter.get(op_name, 0)
        self._counter[op_name] = idx + 1
        return idx

    def run_op(
        self,
        op_name: str,
        inputs: List[Tensor],
        weights: List[Tensor] = None,
        qtype: str = "float32",
        seed: int = None,
        note: str = "",
        **kwargs,
    ) -> Tensor:
        """
        执行单个算子

        Args:
            op_name: 算子名称
            inputs: 输入张量列表
            weights: 权重张量列表
            qtype: 量化类型
            seed: 随机种子
            note: 备注
            **kwargs: 其他参数

        Returns:
            输出 Tensor
        """
        config = get_config()
        op_id = self._get_op_id(op_name)
        weights = weights or []

        # 准备参数
        # 1. 纯 fp32 模式 - 使用原始 fp32 数据
        args_pure = [inp.fp32 for inp in inputs] + [w.fp32 for w in weights]

        # 2. 量化感知模式 - 先量化再反量化
        if config.precision == "quant" and qtype != "float32":
            args_quant = []
            for inp in inputs:
                inp_q = inp.to_qtype(qtype).quantize_dequantize()
                args_quant.append(inp_q.fp32)
            for w in weights:
                w_q = w.to_qtype(qtype).quantize_dequantize()
                args_quant.append(w_q.fp32)
        else:
            args_quant = args_pure

        # 执行 golden
        golden_pure = run_golden(op_name, *args_pure, **kwargs)
        if golden_pure.dtype != np.float32:
            golden_pure = golden_pure.astype(np.float32)

        # 检查是否需要执行量化感知版本
        need_quant_golden = config.precision == "quant" and qtype != "float32"
        if need_quant_golden:
            golden_quant = run_golden(op_name, *args_quant, **kwargs)
            if golden_quant.dtype != np.float32:
                golden_quant = golden_quant.astype(np.float32)
        else:
            golden_quant = golden_pure

        # 创建输出 Tensor
        output = Tensor.from_fp32(golden_quant if config.precision == "quant" else golden_pure, qtype)

        # 记录
        record = OpRecord(
            id=op_id,
            op_name=op_name,
            qtype=qtype,
            inputs=inputs,
            weights=weights,
            output=output,
            golden_pure=golden_pure,
            golden_quant=golden_quant,
            seed=seed,
            note=note,
        )
        self.records.append(record)

        logger.debug(f"执行 {op_name}_{op_id}: {[inp.shape for inp in inputs]} -> {output.shape}")

        return output

    def get_records(self) -> List[OpRecord]:
        """获取所有记录"""
        return self.records

    def dump(self, output_dir: str, format: str = "raw"):
        """
        导出所有记录

        Args:
            output_dir: 输出目录
            format: 数据格式
        """
        from aidevtools.formats.base import save as save_data

        path = Path(output_dir)
        path.mkdir(parents=True, exist_ok=True)

        for r in self.records:
            name = f"{r.op_name}_{r.id}"

            # 保存 golden (纯 fp32)
            if r.golden_pure is not None:
                save_data(str(path / f"{name}_golden_pure.bin"), r.golden_pure, format=format)

            # 保存 golden (量化感知)
            if r.golden_quant is not None:
                save_data(str(path / f"{name}_golden_quant.bin"), r.golden_quant, format=format)

            # 保存输入
            for i, inp in enumerate(r.inputs):
                suffix = f"_input{i}" if len(r.inputs) > 1 else "_input"
                save_data(str(path / f"{name}{suffix}_fp32.bin"), inp.fp32, format=format)
                if inp.quantized is not None:
                    save_data(str(path / f"{name}{suffix}_q.bin"), inp.quantized, format=format)

            # 保存权重
            for i, w in enumerate(r.weights):
                suffix = f"_weight{i}" if len(r.weights) > 1 else "_weight"
                save_data(str(path / f"{name}{suffix}_fp32.bin"), w.fp32, format=format)
                if w.quantized is not None:
                    save_data(str(path / f"{name}{suffix}_q.bin"), w.quantized, format=format)

            logger.info(f"dump: {name}")


# 全局引擎实例
_engine: Optional[ExecutionEngine] = None


def get_engine() -> ExecutionEngine:
    """获取全局引擎"""
    global _engine
    if _engine is None:
        _engine = ExecutionEngine()
    return _engine


def clear():
    """清空记录"""
    get_engine().clear()


def run_op(op_name: str, inputs: List[Tensor], weights: List[Tensor] = None,
           qtype: str = "float32", **kwargs) -> Tensor:
    """执行算子"""
    return get_engine().run_op(op_name, inputs, weights, qtype, **kwargs)


def get_records() -> List[OpRecord]:
    """获取记录"""
    return get_engine().get_records()


def dump(output_dir: str, format: str = "raw"):
    """导出记录"""
    get_engine().dump(output_dir, format)
