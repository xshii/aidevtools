"""算子数据生成器

基于 @register_op 的 auto_gen 配置自动生成测试数据，并管理 L2 内存布局。

数据流:
    1. datagen 生成数据 → 放入 L2 (初始位置)
    2. memory_plan 生成 DMA → L2 搬运到 L1

依赖关系:
    core.memory_types (基础类型)
    └── ops.datagen (本模块)
    不依赖 optimizer 模块

用法:
    from aidevtools.ops.datagen import OpDataGenerator

    gen = OpDataGenerator(seed=42, alignment=256, l2_base=0x100000)

    # 为 linear 算子生成数据 (自动放入 L2)
    data = gen.generate("linear", input_shape=(512, 768), out_features=3072)

    # 查看 L2 内存布局
    print(gen.memory_layout())

    # 导出为 DUT 格式
    gen.export_dut("golden/")
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, TYPE_CHECKING

import numpy as np

from aidevtools.core.random import RandomGenerator
from aidevtools.ops._op_registry import get_op_meta
from aidevtools.ops.registry import get_op_meta as get_meta

# 类型检查时导入（避免运行时循环依赖）
if TYPE_CHECKING:
    from aidevtools.optimizer.memory_plan import MemoryPlan


@dataclass
class TensorInfo:
    """Tensor 信息 (含 L2 内存布局)"""

    name: str
    data: np.ndarray
    shape: Tuple[int, ...]
    dtype: str = "fp32"
    qtype: str = "bfp16"  # 量化类型
    # L2 内存布局
    l2_offset: int = 0  # L2 起始偏移 (字节)
    l2_addr: int = 0  # L2 绝对地址 (字节)
    size: int = 0  # 大小 (字节)
    # 元信息
    strategy: str = ""  # 生成策略
    role: str = "input"  # input / weight / output
    op_name: str = ""  # 所属算子名


@dataclass
class L2MemoryLayout:
    """L2 内存布局"""

    tensors: List[TensorInfo] = field(default_factory=list)
    alignment: int = 256  # 对齐字节数 (256 bytes)
    l2_base: int = 0x100000  # L2 基地址
    current_offset: int = 0  # 当前偏移

    def add(self, info: TensorInfo) -> int:
        """添加 tensor 到 L2，返回分配的绝对地址"""
        # 对齐到 alignment
        aligned_offset = self._align(self.current_offset)
        info.l2_offset = aligned_offset
        info.l2_addr = self.l2_base + aligned_offset
        info.size = info.data.nbytes
        self.tensors.append(info)
        self.current_offset = aligned_offset + info.size
        return info.l2_addr

    def _align(self, offset: int) -> int:
        """对齐到 alignment 边界 (256 bytes)"""
        if offset % self.alignment == 0:
            return offset
        return ((offset // self.alignment) + 1) * self.alignment

    @property
    def total_size(self) -> int:
        """总使用大小"""
        return self.current_offset

    def get_by_name(self, name: str) -> Optional[TensorInfo]:
        """按名称查找 tensor"""
        for t in self.tensors:
            if t.name == name:
                return t
        return None

    def get_by_role(self, role: str) -> List[TensorInfo]:
        """按角色查找 tensor"""
        return [t for t in self.tensors if t.role == role]

    def summary(self) -> str:
        """生成摘要"""
        lines = [
            f"L2 MemoryLayout (base=0x{self.l2_base:X}, alignment={self.alignment} bytes):",
            f"  Total size: {self.total_size} bytes ({self.total_size / 1024 / 1024:.2f} MB)",
            f"  Address range: 0x{self.l2_base:08X} - 0x{self.l2_base + self.total_size:08X}",
            "",
            "  Tensors:",
            f"  {'Name':<25} {'L2 Addr':<12} {'Size':>10} {'Shape':<20} {'QType':<8} Role",
            "  " + "-" * 90,
        ]
        for t in self.tensors:
            role_flag = f"[{t.role[0].upper()}]"  # [I]nput, [W]eight, [O]utput
            lines.append(
                f"  {t.name:<25} 0x{t.l2_addr:08X}  {t.size:>10d} {str(t.shape):<20} {t.qtype:<8} {role_flag}"
            )
        return "\n".join(lines)


class OpDataGenerator:
    """
    算子数据生成器

    基于 @register_op 的 auto_gen 配置自动生成数据，放入 L2 内存。
    """

    def __init__(
        self,
        seed: Optional[int] = None,
        alignment: int = 256,
        l2_base: int = 0x100000,
        dtype: str = "fp32",
        qtype: str = "bfp16",
    ):
        """
        Args:
            seed: 随机种子
            alignment: 内存对齐字节数 (默认 256)
            l2_base: L2 内存基地址
            dtype: 默认数据类型 (fp32)
            qtype: 默认量化类型 (bfp16)
        """
        self.seed = seed
        self.alignment = alignment
        self.l2_base = l2_base
        self.dtype = dtype
        self.qtype = qtype
        self._rand = RandomGenerator(seed)
        self._layout = L2MemoryLayout(alignment=alignment, l2_base=l2_base)
        self._generated: Dict[str, TensorInfo] = {}
        self._op_counter: Dict[str, int] = {}  # 算子计数器

    def reset(self, seed: Optional[int] = None):
        """重置生成器"""
        if seed is not None:
            self.seed = seed
        self._rand.reset(self.seed)
        self._layout = L2MemoryLayout(alignment=self.alignment, l2_base=self.l2_base)
        self._generated = {}
        self._op_counter = {}

    def generate(
        self,
        op_name: str,
        input_shape: Tuple[int, ...],
        qtypes: Optional[Dict[str, str]] = None,
        **kwargs,
    ) -> Dict[str, TensorInfo]:
        """
        为指定算子生成数据，放入 L2 内存

        Args:
            op_name: 算子名称
            input_shape: 主输入 shape
            qtypes: 各参数的量化类型 {"input": "bfp16", "weight": "bfp8", ...}
            **kwargs: 额外参数 (如 out_features, num_heads 等)

        Returns:
            Dict[param_name, TensorInfo]

        Example:
            gen = OpDataGenerator(seed=42, l2_base=0x100000)

            # 统一量化类型
            data = gen.generate("linear", input_shape=(512, 768), out_features=3072)

            # 不同参数不同量化类型
            data = gen.generate(
                "linear",
                input_shape=(512, 768),
                out_features=3072,
                qtypes={"input": "bfp16", "weight": "bfp8", "bias": "fp32"}
            )
        """
        meta = get_meta(op_name)
        if meta is None:
            raise ValueError(f"未注册的算子: {op_name}")

        qtypes = qtypes or {}

        # 算子实例编号 (支持多次调用同一算子)
        if op_name not in self._op_counter:
            self._op_counter[op_name] = 0
        op_idx = self._op_counter[op_name]
        self._op_counter[op_name] += 1
        op_instance_name = f"{op_name}_{op_idx}" if op_idx > 0 else op_name

        result = {}
        shapes_context = {"input_shape": input_shape, **kwargs}

        # 按 auto_gen 配置生成每个参数
        for param, strategy in meta.auto_gen.items():
            is_weight = param in meta.weight_params
            role = "weight" if is_weight else "input"

            # 获取该参数的量化类型 (优先用 qtypes 指定，否则用默认)
            param_qtype = qtypes.get(param, self.qtype)

            data, shape = self._generate_param(
                param, strategy, shapes_context, is_weight
            )

            # 唯一名称
            tensor_name = f"{op_instance_name}.{param}"

            info = TensorInfo(
                name=tensor_name,
                data=data,
                shape=shape,
                dtype=self.dtype,
                qtype=param_qtype,
                strategy=strategy,
                role=role,
                op_name=op_instance_name,
            )

            # 分配 L2 内存 (256 字节对齐)
            self._layout.add(info)
            result[param] = info
            self._generated[tensor_name] = info

            # 更新 context (后续参数可能依赖前面的 shape)
            shapes_context[f"{param}_shape"] = shape
            shapes_context[param] = data

        return result

    def _generate_param(
        self,
        param: str,
        strategy: str,
        context: Dict[str, Any],
        is_weight: bool,
    ) -> Tuple[np.ndarray, Tuple[int, ...]]:
        """根据策略生成参数（委托给 RandomGenerator）"""
        return self._rand.generate_from_strategy(strategy, context)

    def generate_batch(
        self,
        op_configs: List[Dict[str, Any]],
    ) -> Dict[str, Dict[str, TensorInfo]]:
        """
        批量生成多个算子的数据

        Args:
            op_configs: 算子配置列表

        Returns:
            Dict[op_name, Dict[param_name, TensorInfo]]

        Example:
            configs = [
                {"op_name": "linear", "input_shape": (512, 768), "out_features": 3072},
                {"op_name": "gelu", "input_shape": (512, 3072)},
                {"op_name": "linear", "input_shape": (512, 3072), "out_features": 768},
            ]
            batch = gen.generate_batch(configs)
        """
        results = {}
        for i, cfg in enumerate(op_configs):
            op_name = cfg.pop("op_name")
            key = f"{op_name}_{i}"
            results[key] = self.generate(op_name, **cfg)
        return results

    def memory_layout(self) -> L2MemoryLayout:
        """获取 L2 内存布局"""
        return self._layout

    def create_memory_plan(
        self,
        l1_size: int = 256 * 1024,
        l2_size: int = 2 * 1024 * 1024,
    ) -> "MemoryPlan":
        """
        创建 MemoryPlan (用于 DMA 生成)

        注意: 此方法会延迟导入 optimizer.memory_plan 模块。
        如果不需要 DMA 规划，可以不调用此方法，避免引入 optimizer 依赖。

        Args:
            l1_size: L1 大小
            l2_size: L2 大小

        Returns:
            MemoryPlan 对象
        """
        # 延迟导入，避免模块级循环依赖
        from aidevtools.optimizer.memory_plan import MemoryPlan
        from aidevtools.core.memory_types import MemoryLevel

        plan = MemoryPlan(
            l1_size=l1_size,
            l2_size=l2_size,
            l2_base=self.l2_base,
        )

        # 将已生成的 tensor 注册到 plan
        for info in self._layout.tensors:
            plan.allocate_tensor(
                name=info.name.split(".")[-1],  # 去掉 op 前缀
                op_name=info.op_name,
                role=info.role,
                shape=info.shape,
                dtype=info.qtype,  # 用量化类型
                levels=[MemoryLevel.L2],  # 初始只在 L2
            )

        return plan

    def export_dut(
        self,
        output_dir: Union[str, Path],
        prefix: str = "",
    ) -> Dict[str, Path]:
        """
        导出为 DUT 格式 (使用各 tensor 自己的 qtype)

        Args:
            output_dir: 输出目录
            prefix: 文件名前缀

        Returns:
            Dict[tensor_name, file_path]
        """
        from aidevtools.formats import save
        from aidevtools.formats.quantize import quantize

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        result = {}
        for info in self._layout.tensors:
            # 用 tensor 自己的 qtype 量化
            packed, meta = quantize(info.data, info.qtype)

            # 保存 (文件名用 . 替换为 _)
            safe_name = info.name.replace(".", "_")
            filename = f"{prefix}{safe_name}.bin" if prefix else f"{safe_name}.bin"
            filepath = output_dir / filename
            save(str(filepath), packed, fmt="raw")

            result[info.name] = filepath

        # 保存内存布局信息
        layout_file = output_dir / f"{prefix}memory_layout.txt"
        layout_file.write_text(self._layout.summary())

        # 保存量化类型信息
        qtype_file = output_dir / f"{prefix}qtypes.txt"
        qtype_lines = [f"{t.name}: {t.qtype}" for t in self._layout.tensors]
        qtype_file.write_text("\n".join(qtype_lines))

        return result

    def export_header(
        self,
        output_path: Union[str, Path],
        prefix: str = "DATA",
    ) -> Path:
        """
        导出 C 头文件 (包含 L2 内存地址和量化类型)

        Args:
            output_path: 输出文件路径
            prefix: 宏定义前缀

        Returns:
            文件路径
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        lines = [
            "/* Auto-generated L2 memory layout header */",
            f"/* L2 Base: 0x{self.l2_base:X}, Alignment: {self.alignment} bytes */",
            "",
            "#ifndef _MEMORY_LAYOUT_H_",
            "#define _MEMORY_LAYOUT_H_",
            "",
            f"#define {prefix}_L2_BASE 0x{self.l2_base:08X}",
            f"#define {prefix}_ALIGNMENT {self.alignment}",
            f"#define {prefix}_TOTAL_SIZE {self._layout.total_size}",
            "",
            "/* Quantization type codes */",
            "#define QTYPE_FP32    0",
            "#define QTYPE_FP16    1",
            "#define QTYPE_BFP16   2",
            "#define QTYPE_BFP8    3",
            "#define QTYPE_BFP4    4",
            "#define QTYPE_GFLOAT16 5",
            "#define QTYPE_GFLOAT8  6",
            "",
        ]

        qtype_map = {
            "fp32": "QTYPE_FP32",
            "fp16": "QTYPE_FP16",
            "bfp16": "QTYPE_BFP16",
            "bfp8": "QTYPE_BFP8",
            "bfp4": "QTYPE_BFP4",
            "gfloat16": "QTYPE_GFLOAT16",
            "gfloat8": "QTYPE_GFLOAT8",
        }

        for info in self._layout.tensors:
            # 安全的宏名称 (替换 . 为 _)
            safe_name = info.name.replace(".", "_").upper()
            qtype_code = qtype_map.get(info.qtype, "QTYPE_FP32")

            lines.append(f"/* {info.name}: {info.role}, shape={info.shape}, qtype={info.qtype} */")
            lines.append(f"#define {prefix}_{safe_name}_L2_ADDR 0x{info.l2_addr:08X}")
            lines.append(f"#define {prefix}_{safe_name}_SIZE {info.size}")
            lines.append(f"#define {prefix}_{safe_name}_QTYPE {qtype_code}")
            shape_str = ", ".join(str(d) for d in info.shape)
            lines.append(f"#define {prefix}_{safe_name}_SHAPE {{{shape_str}}}")
            lines.append("")

        lines.append("#endif /* _MEMORY_LAYOUT_H_ */")
        lines.append("")

        output_path.write_text("\n".join(lines))
        return output_path


# ============================================================
# 便捷函数
# ============================================================


def generate_op_data(
    op_name: str,
    input_shape: Tuple[int, ...],
    seed: Optional[int] = None,
    alignment: int = 256,
    **kwargs,
) -> Tuple[Dict[str, TensorInfo], L2MemoryLayout]:
    """
    便捷函数: 为算子生成数据

    Returns:
        (data_dict, memory_layout)
    """
    gen = OpDataGenerator(seed=seed, alignment=alignment)
    data = gen.generate(op_name, input_shape, **kwargs)
    return data, gen.memory_layout()


def generate_and_export(
    op_name: str,
    input_shape: Tuple[int, ...],
    output_dir: Union[str, Path],
    qtype: str = "bfp16",
    seed: Optional[int] = None,
    alignment: int = 256,
    **kwargs,
) -> Dict[str, Path]:
    """
    便捷函数: 生成数据并导出为 DUT 格式

    Returns:
        Dict[param_name, file_path]
    """
    gen = OpDataGenerator(seed=seed, alignment=alignment, qtype=qtype)
    gen.generate(op_name, input_shape, **kwargs)
    return gen.export_dut(output_dir)


def generate_with_golden(
    op_name: str,
    input_shape: Tuple[int, ...],
    seed: Optional[int] = None,
    alignment: int = 256,
    **kwargs,
) -> Tuple[Dict[str, TensorInfo], np.ndarray, L2MemoryLayout]:
    """
    便捷函数: 生成数据并计算 golden 输出

    Args:
        op_name: 算子名称
        input_shape: 主输入 shape
        seed: 随机种子
        alignment: 内存对齐字节数
        **kwargs: 额外参数 (如 out_features)

    Returns:
        (data_dict, golden_output, memory_layout)

    Example:
        data, golden, layout = generate_with_golden(
            "linear",
            input_shape=(512, 768),
            out_features=3072,
            seed=42,
        )
        print(f"输入: {data['input'].shape}")
        print(f"权重: {data['weight'].shape}")
        print(f"golden: {golden.shape}")
    """
    from aidevtools.ops.registry import get_op_instance, get_op_meta

    gen = OpDataGenerator(seed=seed, alignment=alignment)
    data = gen.generate(op_name, input_shape, **kwargs)

    # 获取算子实例
    op = get_op_instance(op_name)
    if op is None:
        raise ValueError(f"算子 '{op_name}' 未注册或无法实例化")

    # 获取元信息
    meta = get_op_meta(op_name)
    if meta is None:
        raise ValueError(f"算子 '{op_name}' 无元信息")

    # 构建参数
    args = []
    kwargs_call = {}
    for inp in meta.inputs:
        if inp in data:
            args.append(data[inp].data)
    for opt in meta.optional:
        if opt in data:
            kwargs_call[opt] = data[opt].data

    # 计算 golden
    golden = op.cpu_golden(*args, **kwargs_call)

    return data, golden, gen.memory_layout()
