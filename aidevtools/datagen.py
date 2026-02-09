"""统一数据生成器

单一入口，支持：
- 根据 @register_op 的 auto_gen 自动生成数据
- L2 内存布局管理 (256 字节对齐)
- Golden 计算（四种比数模式）
- DUT 格式导出
- 算子级精度配置 (PrecisionConfig)
- 量化感知随机数生成 (qa_aware 开关)

四种比数模式:
    Track 1: golden_pure   — 纯 fp32 计算的 golden（基准参考）
    Track 2: golden_local  — 本地格式 (fp16/int16/int8) 原生数据计算 golden
                             fp16/bf16: 直接 cast 到目标精度 (原生 fp16 数据)
                             int8/int16: 直接生成目标格式的随机整数
    Track 3: golden_hw     — 硬件格式 (bfp/gfp) 量化→反量化后的模糊权重计算 golden
    Track 4: golden_qa     — 量化感知随机权重计算 golden（受控动态范围）

用法 1: DataGenerator (显式生成)
    from aidevtools import DataGenerator

    gen = DataGenerator(seed=42)
    data = gen.generate("linear", input_shape=(512, 768), out_features=3072)
    gen.export("./golden/")

    # 带精度配置
    from aidevtools.frontend.types import PrecisionConfig
    pc = PrecisionConfig(input_dtype="fp16", weight_dtype="int8",
                         compute_dtype="fp32", output_dtype="bfp8",
                         qa_aware=True)
    tracks = gen.generate_four_track("linear", input_shape=(512, 768),
                                      precision=pc, out_features=3072)

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

用法 3: Model DSL 量化感知模式
    from aidevtools import Model
    from aidevtools.frontend.types import PrecisionConfig

    pc = PrecisionConfig(input_dtype="fp16", compute_dtype="fp32",
                         output_dtype="bfp8", qa_aware=True)
    with Model(seed=42, precision=pc) as m:
        x = m.input((512, 768))
        y = m.linear(x, out_features=3072)
        tracks = m.get_four_track_golden()
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


@dataclass
class FourTrackGolden:
    """四种比数的 golden 输出

    Track 1: golden_pure   — 纯 fp32 计算的 golden
    Track 2: golden_local  — 本地格式模糊权重的 golden
    Track 3: golden_hw     — 硬件格式模糊权重的 golden
    Track 4: golden_qa     — 量化感知随机权重的 golden
    """
    golden_pure: np.ndarray                      # Track 1: 纯 fp32
    golden_local: Optional[np.ndarray] = None    # Track 2: 本地格式模糊
    golden_hw: Optional[np.ndarray] = None       # Track 3: 硬件格式模糊
    golden_qa: Optional[np.ndarray] = None       # Track 4: 量化感知

    # 生成这些 golden 时使用的数据
    data_pure: Optional[Dict[str, np.ndarray]] = None
    data_local: Optional[Dict[str, np.ndarray]] = None
    data_hw: Optional[Dict[str, np.ndarray]] = None
    data_qa: Optional[Dict[str, np.ndarray]] = None

    @property
    def all_goldens(self) -> Dict[str, np.ndarray]:
        """返回所有非 None 的 golden"""
        result = {"pure": self.golden_pure}
        if self.golden_local is not None:
            result["local"] = self.golden_local
        if self.golden_hw is not None:
            result["hw"] = self.golden_hw
        if self.golden_qa is not None:
            result["qa"] = self.golden_qa
        return result


def _to_native_local_dtype(data_fp32: np.ndarray, local_dtype: str, rng=None) -> np.ndarray:
    """生成目标本地格式的原生数据。

    直接在目标格式生成，不做 fp32→target→fp32 往返:
    - fp16/bf16: 直接 cast 到 fp16 (numpy 原生精度)
    - int8/int16: 直接生成同 shape 的随机整数 (原生整型数据)
    - fp32/其他: 原样返回

    返回的数据保持目标原生 dtype (fp16/int8/int16 等)。
    用于 golden 计算时应调用 .astype(np.float32)。

    Args:
        data_fp32: fp32 源数据 (用于确定 shape 及浮点 cast)
        local_dtype: 目标本地格式 ("fp16"/"bf16"/"int8"/"int16"/"fp32")
        rng: numpy Generator 实例，用于整型数据生成 (可选)
    """
    float_map = {
        "fp16": np.float16, "float16": np.float16,
        "bf16": np.float16,  # numpy 不直接支持 bf16，用 fp16 近似
    }
    int_map = {"int8": np.int8, "int16": np.int16}

    if local_dtype in float_map:
        # 浮点本地格式: 直接 cast 到目标精度
        return data_fp32.astype(float_map[local_dtype])
    elif local_dtype in int_map:
        # 整型本地格式: 直接生成目标类型的随机整数
        target = int_map[local_dtype]
        max_val = np.iinfo(target).max
        if rng is not None:
            return rng.integers(-max_val, max_val + 1, size=data_fp32.shape, dtype=target)
        else:
            return np.random.default_rng().integers(
                -max_val, max_val + 1, size=data_fp32.shape, dtype=target
            )
    else:
        return data_fp32


class DataGenerator:
    """
    统一数据生成器

    合并了 frontend/DataGenerator 和 ops/OpDataGenerator 的功能。
    支持:
    - 算子级精度配置 (PrecisionConfig)
    - 量化感知随机数 (qa_aware 开关)
    - 四种比数 golden 生成
    """

    def __init__(
        self,
        seed: Optional[int] = None,
        l2_base: int = 0x100000,
        alignment: int = 256,
        qtype: str = "bfp16",
        precision: Optional[Any] = None,
    ):
        """
        Args:
            seed: 随机种子
            l2_base: L2 内存基地址
            alignment: 内存对齐 (默认 256 字节)
            qtype: 默认量化类型
            precision: PrecisionConfig，算子级精度配置
        """
        import datetime

        self.seed = seed
        self.l2_base = l2_base
        self.alignment = alignment
        self.qtype = qtype
        self._rand = RandomGenerator(seed)
        self._l2_offset = 0
        self._tensors: Dict[str, GeneratedTensor] = {}
        self._op_counter: Dict[str, int] = {}

        # 生成时间戳版本号 (YYYYMMDDHHMMSS)
        self._version = datetime.datetime.now().strftime("%Y%m%d%H%M%S")

        # 精度配置
        if precision is None:
            from aidevtools.frontend.types import PrecisionConfig
            self.precision = PrecisionConfig()
        else:
            self.precision = precision

        # 同步全局 QA 配置到 RandomGenerator
        self._sync_qa_config()

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
        precision: Optional[Any] = None,
        **kwargs,
    ) -> Dict[str, GeneratedTensor]:
        """
        根据 @register_op 的 auto_gen 配置自动生成数据

        Args:
            op_name: 算子名称
            input_shape: 主输入 shape
            qtypes: 各参数的量化类型 {"input": "bfp16", "weight": "bfp8"}
            precision: 可选的算子级 PrecisionConfig，覆盖全局配置
            **kwargs: 额外参数 (out_features, num_heads 等)

        Returns:
            Dict[param_name, GeneratedTensor]
        """
        from aidevtools.ops.registry import get_op_meta

        meta = get_op_meta(op_name)
        if meta is None:
            raise ValueError(f"算子 '{op_name}' 未注册")

        pc = precision or self.precision
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

            # 确定该参数的量化类型 (逐参数精度覆盖)
            if param in qtypes:
                param_qtype = qtypes[param]
            else:
                param_qtype = pc.get_dtype(param, is_weight)
                if param_qtype == "fp32":
                    param_qtype = self.qtype

            # 生成数据 (QA 配置已通过全局设置生效)
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

    def generate_four_track(
        self,
        op_name: str,
        input_shape: Tuple[int, ...],
        precision: Optional[Any] = None,
        **kwargs,
    ) -> FourTrackGolden:
        """生成四种比数的 golden 数据。

        四种比数:
            Track 1: golden_pure   — 纯 fp32 计算
            Track 2: golden_local  — 本地格式 (fp16/int16/int8) 量化→反量化
            Track 3: golden_hw     — 硬件格式 (bfp/gfp) 量化→反量化
            Track 4: golden_qa     — 量化感知随机权重

        Args:
            op_name: 算子名称
            input_shape: 输入 shape
            precision: PrecisionConfig (可选, 覆盖全局)
            **kwargs: 额外参数

        Returns:
            FourTrackGolden 包含四种 golden 及其对应的输入数据
        """
        from aidevtools.ops.registry import get_op_instance, get_op_meta
        from aidevtools.formats.quantize import simulate_quantize

        pc = precision or self.precision
        meta = get_op_meta(op_name)
        if meta is None:
            raise ValueError(f"算子 '{op_name}' 未注册")

        op = get_op_instance(op_name)
        if op is None:
            raise ValueError(f"算子 '{op_name}' 无法实例化")

        # ---------- Track 1: 纯 fp32 golden ----------
        # 保存和恢复生成器状态用于复现
        saved_seed = self.seed
        self._rand.reset(saved_seed)

        context = {"input_shape": input_shape, **kwargs}
        pure_data = {}
        for param, strategy in meta.auto_gen.items():
            array, shape = self._rand.generate_from_strategy(strategy, context)
            pure_data[param] = array.astype(np.float32)
            context[f"{param}_shape"] = shape
            context[param] = array

        # 构建调用参数 — 处理非 auto_gen 的 inputs (如 normalized_shape)
        pure_args = []
        for inp in meta.inputs:
            if inp in pure_data:
                pure_args.append(pure_data[inp])
            elif inp == "normalized_shape":
                pure_args.append((input_shape[-1],))
            elif inp in kwargs:
                pure_args.append(kwargs[inp])
        pure_kwargs = {opt: pure_data[opt] for opt in meta.optional if opt in pure_data}
        golden_pure = op.cpu_golden(*pure_args, **pure_kwargs)

        # ---------- Track 2: 本地格式原生数据 golden ----------
        # 直接在目标格式生成数据，不经过 fp32→target→fp32 中转
        golden_local = None
        local_data = None
        local_dtypes = ("fp16", "float16", "bf16", "int16", "int8")
        has_local = any(
            pc.get_dtype(p, p in meta.weight_params) in local_dtypes
            for p in pure_data
        )
        if has_local:
            local_data = {}
            for param, arr in pure_data.items():
                is_weight = param in meta.weight_params
                dtype_to_use = pc.get_dtype(param, is_weight)
                local_data[param] = _to_native_local_dtype(
                    arr, dtype_to_use, rng=self._rand._rng,
                )

            # Golden 计算: cast 到 fp32
            local_args = []
            for inp in meta.inputs:
                if inp in local_data:
                    local_args.append(local_data[inp].astype(np.float32))
                elif inp == "normalized_shape":
                    local_args.append((input_shape[-1],))
                elif inp in kwargs:
                    local_args.append(kwargs[inp])
            local_kwargs = {
                opt: local_data[opt].astype(np.float32)
                for opt in meta.optional if opt in local_data
            }
            golden_local = op.cpu_golden(*local_args, **local_kwargs)

        # ---------- Track 3: 硬件格式模糊权重 golden ----------
        hw_dtype = pc.output_dtype  # 使用 output_dtype 确定硬件格式
        golden_hw = None
        hw_data = None
        hw_qtypes = ("bfp16", "bfp8", "bfp4", "gfloat16", "gfloat8", "gfloat4",
                     "gfp16", "gfp8", "gfp4")
        # 检查是否有任何精度配置使用了硬件格式
        all_dtypes = {getattr(pc, f) for f in ("input_dtype", "weight_dtype", "output_dtype")}
        all_dtypes.update(pc.param_dtypes.values())
        has_hw = bool(all_dtypes & set(hw_qtypes))
        if has_hw:
            hw_data = {}
            for param, arr in pure_data.items():
                is_weight = param in meta.weight_params
                dtype_to_use = pc.get_dtype(param, is_weight)
                if dtype_to_use in hw_qtypes:
                    hw_data[param] = simulate_quantize(arr, dtype_to_use)
                else:
                    hw_data[param] = arr.copy()

            hw_args = []
            for inp in meta.inputs:
                if inp in hw_data:
                    hw_args.append(hw_data[inp])
                elif inp == "normalized_shape":
                    hw_args.append((input_shape[-1],))
                elif inp in kwargs:
                    hw_args.append(kwargs[inp])
            hw_kwargs = {opt: hw_data[opt] for opt in meta.optional if opt in hw_data}
            golden_hw = op.cpu_golden(*hw_args, **hw_kwargs)

        # ---------- Track 4: 量化感知随机权重 golden ----------
        golden_qa = None
        qa_data = None
        if pc.qa_aware:
            self._rand.reset(saved_seed)
            # 临时启用 QA 全局配置
            self._rand.set_qa_config(
                enabled=True,
                center=pc.qa_center,
                amplitude=pc.qa_amplitude,
            )
            qa_context = {"input_shape": input_shape, **kwargs}
            qa_data = {}
            for param, strategy in meta.auto_gen.items():
                array, shape = self._rand.generate_from_strategy(
                    strategy, qa_context,
                )
                qa_data[param] = array.astype(np.float32)
                qa_context[f"{param}_shape"] = shape
                qa_context[param] = array

            qa_args = []
            for inp in meta.inputs:
                if inp in qa_data:
                    qa_args.append(qa_data[inp])
                elif inp == "normalized_shape":
                    qa_args.append((input_shape[-1],))
                elif inp in kwargs:
                    qa_args.append(kwargs[inp])
            qa_kwargs = {opt: qa_data[opt] for opt in meta.optional if opt in qa_data}
            golden_qa = op.cpu_golden(*qa_args, **qa_kwargs)
            # 恢复全局配置
            self._sync_qa_config()

        return FourTrackGolden(
            golden_pure=golden_pure,
            golden_local=golden_local,
            golden_hw=golden_hw,
            golden_qa=golden_qa,
            data_pure=pure_data,
            data_local=local_data,
            data_hw=hw_data,
            data_qa=qa_data,
        )

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
        if self.precision.qa_aware:
            array = self._rand.qa_uniform(shape,
                                          center=self.precision.qa_center,
                                          amplitude=self.precision.qa_amplitude)
        else:
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
        if self.precision.qa_aware:
            array = self._rand.qa_uniform(shape,
                                          center=self.precision.qa_center,
                                          amplitude=self.precision.qa_amplitude)
        else:
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
        if self.precision.qa_aware:
            array = self._rand.qa_uniform(shape,
                                          center=self.precision.qa_center,
                                          amplitude=self.precision.qa_amplitude)
        else:
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
        if self.precision.qa_aware:
            array = self._rand.qa_uniform(shape,
                                          center=self.precision.qa_center,
                                          amplitude=self.precision.qa_amplitude)
        else:
            array = self._rand.kaiming(shape)
        name = name or self._auto_name("weight")
        return self._add_tensor(name, array, qtype=qtype or self.qtype, role="weight")

    def qa_randn(
        self,
        shape: Tuple[int, ...],
        center: float = 1.0,
        amplitude: float = 0.5,
        name: Optional[str] = None,
        qtype: Optional[str] = None,
        role: str = "input",
    ) -> GeneratedTensor:
        """量化感知随机数（显式调用）"""
        array = self._rand.qa_uniform(shape, center=center, amplitude=amplitude)
        name = name or self._auto_name("qa_randn")
        return self._add_tensor(name, array, qtype=qtype or self.qtype, role=role)

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
        bm: str = "",
    ) -> Dict[str, Path]:
        """
        导出为 DUT 格式

        文件命名: {bm}_{version}_{name}_{shape}.{qtype}.bin
        示例: encoder_20260208_143025_linear_0_weight_64x64.bfp4.bin

        Args:
            output_dir: 输出目录
            prefix: 文件名前缀 (旧接口，保持兼容)
            bm: benchmark 名称前缀 (如 "encoder")

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

            # 文件名: {bm}_{version}_{name}_{shape}.{qtype}.bin
            safe_name = tensor.name.replace(".", "_")
            shape_suffix = "x".join(str(s) for s in tensor.shape)
            parts = []
            if prefix:
                parts.append(prefix)
            if bm:
                parts.append(bm)
            # 自动添加时间戳版本号
            parts.append(self._version)
            parts.append(safe_name)
            parts.append(shape_suffix)
            basename = "_".join(parts)
            filename = f"{basename}.{tensor.qtype}.bin"
            filepath = output_dir / filename
            save(str(filepath), packed, fmt="raw")
            result[tensor.name] = filepath

        # 保存内存布局
        layout_prefix = f"{bm}_{self._version}_" if bm else f"{self._version}_" if not prefix else f"{prefix}_"
        (output_dir / f"{layout_prefix}memory_layout.txt").write_text(self.memory_summary())

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

    def _sync_qa_config(self):
        """同步 PrecisionConfig 的 QA 配置到 RandomGenerator 全局配置"""
        self._rand.set_qa_config(
            enabled=self.precision.qa_aware,
            center=self.precision.qa_center,
            amplitude=self.precision.qa_amplitude,
        )

    def _gen_from_strategy(
        self,
        strategy: str,
        context: Dict[str, Any],
    ) -> Tuple[np.ndarray, Tuple[int, ...]]:
        """根据 auto_gen 策略生成数据（委托给 RandomGenerator）

        QA 配置已通过 _sync_qa_config 设为全局，无需逐调用传递。
        """
        return self._rand.generate_from_strategy(strategy, context)


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

    支持:
    - 自动权重生成
    - 算子级精度配置 (PrecisionConfig)
    - 量化感知随机数 (qa_aware 开关)
    - 四种比数 golden 生成

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

    量化感知用法:
        from aidevtools.frontend.types import PrecisionConfig

        pc = PrecisionConfig(input_dtype="fp16", compute_dtype="fp32",
                             qa_aware=True, qa_center=1.0, qa_amplitude=0.5)

        with Model(seed=42, precision=pc) as m:
            x = m.input((512, 768))
            y = m.linear(x, out_features=3072)
    """

    def __init__(
        self,
        seed: Optional[int] = None,
        l2_base: int = 0x100000,
        alignment: int = 256,
        qtype: str = "bfp16",
        precision: Optional[Any] = None,
        data_dir: Optional[Union[str, Path]] = None,
        bm: str = "",
    ):
        self._gen = DataGenerator(
            seed=seed, l2_base=l2_base, alignment=alignment,
            qtype=qtype, precision=precision,
        )
        self._outputs: List[ModelTensor] = []
        self._op_counter: Dict[str, int] = {}
        self._data_dir = Path(data_dir) if data_dir else None
        self._bm = bm
        self._loaded: Optional[Dict[str, np.ndarray]] = None
        if self._data_dir:
            from aidevtools.formats.base import load_dir
            self._loaded = load_dir(str(self._data_dir), bm=bm)

    def __enter__(self) -> "Model":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False

    @property
    def precision(self):
        """获取当前精度配置"""
        return self._gen.precision

    # ============================================================
    # 输入
    # ============================================================

    def input(
        self,
        shape: Tuple[int, ...],
        name: Optional[str] = None,
        qtype: Optional[str] = None,
    ) -> ModelTensor:
        """定义输入 (data_dir 模式下从文件加载)"""
        name = name or self._next_name("input")
        key = name.replace(".", "_")
        if self._loaded and key in self._loaded:
            array = self._loaded[key]
            return ModelTensor(shape=array.shape, name=name, golden=array)
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
    # 四种比数
    # ============================================================

    def generate_four_track(
        self,
        op_name: str,
        input_shape: Tuple[int, ...],
        precision: Optional[Any] = None,
        **kwargs,
    ) -> FourTrackGolden:
        """为指定算子生成四种比数 golden (委托给 DataGenerator)"""
        return self._gen.generate_four_track(
            op_name, input_shape, precision=precision, **kwargs
        )

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

        # 处理非 auto_gen 的 inputs (如 normalized_shape)
        for inp in meta.inputs[1:]:
            if inp not in meta.auto_gen:
                if inp == "normalized_shape":
                    args.append((input_shape[-1],))
                elif inp in kwargs:
                    args.append(kwargs[inp])

        kwargs_call = {}

        pc = self._gen.precision
        for param, strategy in meta.auto_gen.items():
            if param == meta.inputs[0]:
                # 第一个输入已经有了
                continue

            is_weight = param in meta.weight_params
            role = "weight" if is_weight else "input"

            # 确定该参数的量化类型 (逐参数精度覆盖)
            param_qtype = pc.get_dtype(param, is_weight)
            if param_qtype == "fp32":
                param_qtype = self._gen.qtype

            # 从 data_dir 加载或生成
            tensor_name = f"{prefix}.{param}"
            load_key = tensor_name.replace(".", "_")
            if self._loaded and load_key in self._loaded:
                array = self._loaded[load_key]
                shape = array.shape
            else:
                array, shape = self._gen._gen_from_strategy(strategy, context)
            self._gen._add_tensor(tensor_name, array, qtype=param_qtype, role=role)

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

    def export(self, output_dir: Union[str, Path], prefix: str = "", bm: str = "") -> Dict[str, Path]:
        """导出为 DUT 格式"""
        return self._gen.export(output_dir, prefix, bm=bm)

    def export_header(self, output_path: Union[str, Path], prefix: str = "DATA") -> Path:
        """导出 C 头文件"""
        return self._gen.export_header(output_path, prefix)
