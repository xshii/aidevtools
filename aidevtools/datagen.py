"""统一数据生成器

单一入口，支持：
- 根据 @register_op 的 auto_gen 自动生成数据
- L2 内存布局管理 (256 字节对齐)
- Golden 计算
- DUT 格式导出

用法 1: DataGenerator (显式生成)
    from aidevtools import DataGenerator

    gen = DataGenerator(seed=42)
    data = gen.generate("linear", input_shape=(512, 768), out_features=3072)
    gen.export("./golden/")

用法 2: Model DSL (自动生成权重)
    from aidevtools import Model

    with Model(seed=42) as m:
        x = m.input((512, 768))
        y = m.linear(x, out_features=3072)  # 自动生成 weight, bias
        y = m.gelu(y)
        y = m.linear(y, out_features=768)

    print(m.tensors)   # 所有生成的数据
    print(m.outputs)   # golden 输出
    m.export("./golden/")
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from aidevtools.core.random import RandomGenerator


@dataclass
class GeneratedTensor:
    """生成的张量"""
    name: str
    array: np.ndarray
    # L2 内存
    l2_addr: int = 0
    l2_size: int = 0
    # 元信息
    qtype: str = "fp32"
    role: str = "input"  # input / weight / output

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.array.shape

    @property
    def dtype(self) -> np.dtype:
        return self.array.dtype

    def __repr__(self):
        return f"GeneratedTensor({self.name}, shape={self.shape}, L2=0x{self.l2_addr:X})"


class DataGenerator:
    """
    统一数据生成器

    合并了 frontend/DataGenerator 和 ops/OpDataGenerator 的功能。
    """

    def __init__(
        self,
        seed: Optional[int] = None,
        l2_base: int = 0x100000,
        alignment: int = 256,
        qtype: str = "bfp16",
    ):
        """
        Args:
            seed: 随机种子
            l2_base: L2 内存基地址
            alignment: 内存对齐 (默认 256 字节)
            qtype: 默认量化类型
        """
        self.seed = seed
        self.l2_base = l2_base
        self.alignment = alignment
        self.qtype = qtype
        self._rand = RandomGenerator(seed)
        self._l2_offset = 0
        self._tensors: Dict[str, GeneratedTensor] = {}
        self._op_counter: Dict[str, int] = {}

    def reset(self, seed: Optional[int] = None):
        """重置生成器"""
        if seed is not None:
            self.seed = seed
        self._rand.reset(self.seed)
        self._l2_offset = 0
        self._tensors.clear()
        self._op_counter.clear()

    # ============================================================
    # 自动生成 (读取 @register_op)
    # ============================================================

    def generate(
        self,
        op_name: str,
        input_shape: Tuple[int, ...],
        qtypes: Optional[Dict[str, str]] = None,
        **kwargs,
    ) -> Dict[str, GeneratedTensor]:
        """
        根据 @register_op 的 auto_gen 配置自动生成数据

        Args:
            op_name: 算子名称
            input_shape: 主输入 shape
            qtypes: 各参数的量化类型 {"input": "bfp16", "weight": "bfp8"}
            **kwargs: 额外参数 (out_features, num_heads 等)

        Returns:
            Dict[param_name, GeneratedTensor]
        """
        from aidevtools.ops.registry import get_op_meta

        meta = get_op_meta(op_name)
        if meta is None:
            raise ValueError(f"算子 '{op_name}' 未注册")

        qtypes = qtypes or {}

        # 算子实例编号
        idx = self._op_counter.get(op_name, 0)
        self._op_counter[op_name] = idx + 1
        prefix = f"{op_name}_{idx}" if idx > 0 else op_name

        result = {}
        context = {"input_shape": input_shape, **kwargs}

        for param, strategy in meta.auto_gen.items():
            is_weight = param in meta.weight_params
            role = "weight" if is_weight else "input"
            param_qtype = qtypes.get(param, self.qtype)

            # 生成数据
            array, shape = self._gen_from_strategy(strategy, context)

            # 创建 tensor
            name = f"{prefix}.{param}"
            tensor = self._add_tensor(name, array, qtype=param_qtype, role=role)
            result[param] = tensor

            # 更新 context
            context[f"{param}_shape"] = shape
            context[param] = array

        return result

    def generate_with_golden(
        self,
        op_name: str,
        input_shape: Tuple[int, ...],
        qtypes: Optional[Dict[str, str]] = None,
        **kwargs,
    ) -> Tuple[Dict[str, GeneratedTensor], np.ndarray]:
        """
        生成数据并计算 golden

        Returns:
            (data_dict, golden_output)
        """
        from aidevtools.ops.registry import get_op_instance, get_op_meta

        data = self.generate(op_name, input_shape, qtypes, **kwargs)

        op = get_op_instance(op_name)
        if op is None:
            raise ValueError(f"算子 '{op_name}' 无法实例化")

        meta = get_op_meta(op_name)

        # 构建参数
        args = [data[inp].array for inp in meta.inputs if inp in data]
        kwargs_call = {opt: data[opt].array for opt in meta.optional if opt in data}

        golden = op.cpu_golden(*args, **kwargs_call)
        return data, golden

    # ============================================================
    # 手动生成
    # ============================================================

    def randn(
        self,
        shape: Tuple[int, ...],
        name: Optional[str] = None,
        qtype: Optional[str] = None,
        role: str = "input",
    ) -> GeneratedTensor:
        """正态分布随机数"""
        array = self._rand.normal(shape)
        name = name or self._auto_name("randn")
        return self._add_tensor(name, array, qtype=qtype or self.qtype, role=role)

    def uniform(
        self,
        shape: Tuple[int, ...],
        low: float = -1.0,
        high: float = 1.0,
        name: Optional[str] = None,
        qtype: Optional[str] = None,
        role: str = "input",
    ) -> GeneratedTensor:
        """均匀分布"""
        array = self._rand.uniform(shape, low=low, high=high)
        name = name or self._auto_name("uniform")
        return self._add_tensor(name, array, qtype=qtype or self.qtype, role=role)

    def zeros(
        self,
        shape: Tuple[int, ...],
        name: Optional[str] = None,
        qtype: Optional[str] = None,
        role: str = "input",
    ) -> GeneratedTensor:
        """全零"""
        array = self._rand.zeros(shape)
        name = name or self._auto_name("zeros")
        return self._add_tensor(name, array, qtype=qtype or self.qtype, role=role)

    def ones(
        self,
        shape: Tuple[int, ...],
        name: Optional[str] = None,
        qtype: Optional[str] = None,
        role: str = "input",
    ) -> GeneratedTensor:
        """全一"""
        array = self._rand.ones(shape)
        name = name or self._auto_name("ones")
        return self._add_tensor(name, array, qtype=qtype or self.qtype, role=role)

    def xavier(
        self,
        shape: Tuple[int, ...],
        name: Optional[str] = None,
        qtype: Optional[str] = None,
    ) -> GeneratedTensor:
        """Xavier 初始化"""
        array = self._rand.xavier(shape)
        name = name or self._auto_name("weight")
        return self._add_tensor(name, array, qtype=qtype or self.qtype, role="weight")

    def kaiming(
        self,
        shape: Tuple[int, ...],
        name: Optional[str] = None,
        qtype: Optional[str] = None,
    ) -> GeneratedTensor:
        """Kaiming 初始化"""
        array = self._rand.kaiming(shape)
        name = name or self._auto_name("weight")
        return self._add_tensor(name, array, qtype=qtype or self.qtype, role="weight")

    # ============================================================
    # L2 内存管理
    # ============================================================

    def _add_tensor(
        self,
        name: str,
        array: np.ndarray,
        qtype: str,
        role: str,
    ) -> GeneratedTensor:
        """添加 tensor 到 L2 内存"""
        # 对齐
        aligned = self._align(self._l2_offset)
        l2_addr = self.l2_base + aligned
        l2_size = array.nbytes

        tensor = GeneratedTensor(
            name=name,
            array=array,
            l2_addr=l2_addr,
            l2_size=l2_size,
            qtype=qtype,
            role=role,
        )

        self._tensors[name] = tensor
        self._l2_offset = aligned + l2_size
        return tensor

    def _align(self, offset: int) -> int:
        """对齐到 alignment 边界"""
        if offset % self.alignment == 0:
            return offset
        return ((offset // self.alignment) + 1) * self.alignment

    @property
    def total_size(self) -> int:
        """总 L2 使用量"""
        return self._l2_offset

    def memory_summary(self) -> str:
        """L2 内存摘要"""
        lines = [
            f"L2 Memory (base=0x{self.l2_base:X}, align={self.alignment})",
            f"Total: {self.total_size / 1024:.1f} KB",
            "",
            f"{'Name':<30} {'Shape':<20} {'L2 Addr':<12} {'Size':>10} {'QType':<8}",
            "-" * 85,
        ]
        for t in self._tensors.values():
            lines.append(
                f"{t.name:<30} {str(t.shape):<20} 0x{t.l2_addr:<10X} {t.l2_size:>10} {t.qtype:<8}"
            )
        return "\n".join(lines)

    # ============================================================
    # 导出
    # ============================================================

    def export(
        self,
        output_dir: Union[str, Path],
        prefix: str = "",
    ) -> Dict[str, Path]:
        """
        导出为 DUT 格式

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
        for tensor in self._tensors.values():
            # 量化
            packed, _ = quantize(tensor.array, tensor.qtype)

            # 保存
            safe_name = tensor.name.replace(".", "_")
            filename = f"{prefix}{safe_name}.bin" if prefix else f"{safe_name}.bin"
            filepath = output_dir / filename
            save(str(filepath), packed, fmt="raw")
            result[tensor.name] = filepath

        # 保存内存布局
        (output_dir / f"{prefix}memory_layout.txt").write_text(self.memory_summary())

        return result

    def export_header(
        self,
        output_path: Union[str, Path],
        prefix: str = "DATA",
    ) -> Path:
        """导出 C 头文件"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        lines = [
            "/* Auto-generated memory layout */",
            f"#ifndef _{prefix}_LAYOUT_H_",
            f"#define _{prefix}_LAYOUT_H_",
            "",
            f"#define {prefix}_L2_BASE 0x{self.l2_base:08X}",
            f"#define {prefix}_TOTAL_SIZE {self.total_size}",
            "",
        ]

        for t in self._tensors.values():
            safe = t.name.replace(".", "_").upper()
            lines.append(f"#define {prefix}_{safe}_ADDR 0x{t.l2_addr:08X}")
            lines.append(f"#define {prefix}_{safe}_SIZE {t.l2_size}")

        lines.append("")
        lines.append(f"#endif /* _{prefix}_LAYOUT_H_ */")

        output_path.write_text("\n".join(lines))
        return output_path

    # ============================================================
    # 内部方法
    # ============================================================

    def _auto_name(self, prefix: str) -> str:
        """自动生成名称"""
        count = sum(1 for n in self._tensors if n.startswith(prefix))
        return f"{prefix}_{count}"

    def _gen_from_strategy(
        self,
        strategy: str,
        context: Dict[str, Any],
    ) -> Tuple[np.ndarray, Tuple[int, ...]]:
        """根据 auto_gen 策略生成数据"""
        input_shape = context.get("input_shape", (1,))

        if strategy == "input":
            shape = input_shape
            data = self._rand.normal(shape)

        elif strategy == "random":
            shape = input_shape
            data = self._rand.normal(shape)

        elif strategy == "xavier":
            out_features = context.get("out_features", input_shape[-1])
            in_features = input_shape[-1]
            shape = (out_features, in_features)
            data = self._rand.xavier(shape)

        elif strategy == "kaiming":
            out_features = context.get("out_features", input_shape[-1])
            in_features = input_shape[-1]
            shape = (out_features, in_features)
            data = self._rand.kaiming(shape)

        elif strategy == "uniform":
            shape = (context.get("out_features", input_shape[-1]),)
            data = self._rand.uniform(shape, low=-0.1, high=0.1)

        elif strategy.startswith("zeros:"):
            shape = self._parse_shape(strategy[6:], context)
            data = self._rand.zeros(shape)

        elif strategy.startswith("ones:"):
            shape = self._parse_shape(strategy[5:], context)
            data = self._rand.ones(shape)

        elif strategy.startswith("xavier:"):
            shape = self._parse_shape(strategy[7:], context)
            data = self._rand.xavier(shape)

        elif strategy.startswith("kaiming:"):
            shape = self._parse_shape(strategy[8:], context)
            data = self._rand.kaiming(shape)

        elif strategy.startswith("same:"):
            ref = strategy[5:]
            shape = context.get(f"{ref}_shape", input_shape)
            data = self._rand.normal(shape)

        else:
            raise ValueError(f"未知策略: {strategy}")

        return data, shape

    def _parse_shape(self, spec: str, context: Dict[str, Any]) -> Tuple[int, ...]:
        """解析 shape 规格"""
        input_shape = context.get("input_shape", (1,))
        parts = [p.strip() for p in spec.split(",") if p.strip()]
        result = []

        for part in parts:
            if part.lstrip("-").isdigit():
                idx = int(part)
                result.append(input_shape[idx] if abs(idx) <= len(input_shape) else 1)
            elif part in context:
                val = context[part]
                result.append(int(val) if isinstance(val, (int, np.integer)) else val)
            else:
                raise ValueError(f"无法解析: {part}")

        return tuple(result)


# ============================================================
# Model DSL - 自动生成权重
# ============================================================


class ModelTensor:
    """DSL 中的 Tensor，记录 shape 用于后续算子推断"""

    def __init__(self, shape: Tuple[int, ...], name: str, golden: Optional[np.ndarray] = None):
        self.shape = shape
        self.name = name
        self.golden = golden

    def __repr__(self):
        return f"ModelTensor({self.name}, shape={self.shape})"


class Model:
    """
    Model DSL - 自动根据 @register_op 生成权重

    用法:
        with Model(seed=42) as m:
            x = m.input((512, 768))
            y = m.linear(x, out_features=3072)  # 自动生成 weight, bias
            y = m.gelu(y)
            y = m.linear(y, out_features=768)

        # 获取所有数据
        for name, t in m.tensors.items():
            print(f"{name}: {t.shape}")

        # 导出
        m.export("./golden/")
    """

    def __init__(
        self,
        seed: Optional[int] = None,
        l2_base: int = 0x100000,
        alignment: int = 256,
        qtype: str = "bfp16",
    ):
        self._gen = DataGenerator(seed=seed, l2_base=l2_base, alignment=alignment, qtype=qtype)
        self._outputs: List[ModelTensor] = []
        self._op_counter: Dict[str, int] = {}

    def __enter__(self) -> "Model":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False

    # ============================================================
    # 输入
    # ============================================================

    def input(
        self,
        shape: Tuple[int, ...],
        name: Optional[str] = None,
        qtype: Optional[str] = None,
    ) -> ModelTensor:
        """定义输入"""
        name = name or self._next_name("input")
        t = self._gen.randn(shape, name=name, qtype=qtype, role="input")
        return ModelTensor(shape=t.shape, name=name, golden=t.array)

    # ============================================================
    # 算子 (自动生成权重)
    # ============================================================

    def linear(self, x: ModelTensor, out_features: int, **kwargs) -> ModelTensor:
        """Linear: 自动生成 weight, bias"""
        return self._call_op("linear", x, out_features=out_features, **kwargs)

    def gelu(self, x: ModelTensor, **kwargs) -> ModelTensor:
        """GELU 激活"""
        return self._call_op("gelu", x, **kwargs)

    def relu(self, x: ModelTensor, **kwargs) -> ModelTensor:
        """ReLU 激活"""
        return self._call_op("relu", x, **kwargs)

    def silu(self, x: ModelTensor, **kwargs) -> ModelTensor:
        """SiLU 激活"""
        return self._call_op("silu", x, **kwargs)

    def softmax(self, x: ModelTensor, **kwargs) -> ModelTensor:
        """Softmax"""
        return self._call_op("softmax", x, **kwargs)

    def layernorm(self, x: ModelTensor, **kwargs) -> ModelTensor:
        """LayerNorm: 自动生成 gamma, beta"""
        return self._call_op("layernorm", x, **kwargs)

    def rmsnorm(self, x: ModelTensor, **kwargs) -> ModelTensor:
        """RMSNorm: 自动生成 gamma"""
        return self._call_op("rmsnorm", x, **kwargs)

    def matmul(self, x: ModelTensor, y: ModelTensor, **kwargs) -> ModelTensor:
        """MatMul (两个输入)"""
        # matmul 特殊处理：两个输入都来自前面的 tensor
        from aidevtools.ops.registry import get_op_instance

        op_name = "matmul"
        prefix = self._next_name(op_name)

        op = get_op_instance(op_name)
        if op is None:
            raise ValueError(f"算子 '{op_name}' 未注册")

        golden = op.cpu_golden(x.golden, y.golden)
        out = ModelTensor(shape=golden.shape, name=f"{prefix}.output", golden=golden)
        self._outputs.append(out)
        return out

    def add(self, x: ModelTensor, y: ModelTensor, **kwargs) -> ModelTensor:
        """Add"""
        from aidevtools.ops.registry import get_op_instance
        op = get_op_instance("add")
        golden = op.cpu_golden(x.golden, y.golden)
        name = self._next_name("add")
        out = ModelTensor(shape=golden.shape, name=f"{name}.output", golden=golden)
        self._outputs.append(out)
        return out

    def mul(self, x: ModelTensor, y: ModelTensor, **kwargs) -> ModelTensor:
        """Mul"""
        from aidevtools.ops.registry import get_op_instance
        op = get_op_instance("mul")
        golden = op.cpu_golden(x.golden, y.golden)
        name = self._next_name("mul")
        out = ModelTensor(shape=golden.shape, name=f"{name}.output", golden=golden)
        self._outputs.append(out)
        return out

    # ============================================================
    # 通用算子调用
    # ============================================================

    def _call_op(self, op_name: str, x: ModelTensor, **kwargs) -> ModelTensor:
        """调用算子，自动生成权重"""
        from aidevtools.ops.registry import get_op_meta, get_op_instance

        meta = get_op_meta(op_name)
        if meta is None:
            raise ValueError(f"算子 '{op_name}' 未注册")

        prefix = self._next_name(op_name)
        input_shape = x.shape

        # 根据 auto_gen 生成权重
        context = {"input_shape": input_shape, **kwargs}
        args = [x.golden]  # 第一个参数是输入
        kwargs_call = {}

        for param, strategy in meta.auto_gen.items():
            if param == meta.inputs[0]:
                # 第一个输入已经有了
                continue

            is_weight = param in meta.weight_params
            role = "weight" if is_weight else "input"

            # 生成数据
            array, shape = self._gen._gen_from_strategy(strategy, context)
            tensor_name = f"{prefix}.{param}"
            self._gen._add_tensor(tensor_name, array, qtype=self._gen.qtype, role=role)

            # 更新 context
            context[f"{param}_shape"] = shape
            context[param] = array

            # 添加到参数
            if param in meta.inputs[1:]:
                args.append(array)
            elif param in meta.optional:
                kwargs_call[param] = array

        # 计算 golden
        op = get_op_instance(op_name)
        golden = op.cpu_golden(*args, **kwargs_call)

        # 返回输出
        out = ModelTensor(shape=golden.shape, name=f"{prefix}.output", golden=golden)
        self._outputs.append(out)
        return out

    def _next_name(self, op_name: str) -> str:
        """生成下一个算子名"""
        idx = self._op_counter.get(op_name, 0)
        self._op_counter[op_name] = idx + 1
        return f"{op_name}_{idx}"

    # ============================================================
    # 访问器
    # ============================================================

    @property
    def tensors(self) -> Dict[str, GeneratedTensor]:
        """所有生成的 tensor (输入 + 权重)"""
        return self._gen._tensors

    @property
    def outputs(self) -> List[ModelTensor]:
        """所有算子的输出"""
        return self._outputs

    @property
    def final_output(self) -> Optional[np.ndarray]:
        """最终输出的 golden"""
        if self._outputs:
            return self._outputs[-1].golden
        return None

    def memory_summary(self) -> str:
        """L2 内存摘要"""
        return self._gen.memory_summary()

    def export(self, output_dir: Union[str, Path], prefix: str = "") -> Dict[str, Path]:
        """导出为 DUT 格式"""
        return self._gen.export(output_dir, prefix)

    def export_header(self, output_path: Union[str, Path], prefix: str = "DATA") -> Path:
        """导出 C 头文件"""
        return self._gen.export_header(output_path, prefix)
