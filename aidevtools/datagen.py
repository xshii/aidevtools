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
        hw_qtypes = ("bfp16", "bfp8", "bfp4",
                     "bfpp16", "bfpp8", "bfpp4",
                     "gfloat16", "gfloat8", "gfloat4",
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
    - CQA 链式量化感知 (cqa 开关)

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

    CQA 链式量化感知:
        pc = PrecisionConfig(input_dtype="bfpp8", weight_dtype="bfpp4",
                             compute_dtype="fp32", output_dtype="bfpp8")

        with Model(seed=42, precision=pc, cqa=True) as m:
            x = m.input((512, 768))       # 输入经过 bfpp8 量化
            y = m.linear(x, out_features=3072)  # 权重 bfpp4 量化, 输出 bfpp8 量化
            y = m.gelu(y)                 # 输出 bfpp8 量化
            y = m.linear(y, out_features=768)

        # 每层输入 = 上层量化后的输出, 模拟硬件真实数据流
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
        cqa: bool = False,
    ):
        self._gen = DataGenerator(
            seed=seed, l2_base=l2_base, alignment=alignment,
            qtype=qtype, precision=precision,
        )
        self._outputs: List[ModelTensor] = []
        self._op_counter: Dict[str, int] = {}
        self._data_dir = Path(data_dir) if data_dir else None
        self._bm = bm
        self._cqa = cqa
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
    # CQA 链式量化
    # ============================================================

    def _cqa_quantize(self, data: np.ndarray, qtype: str) -> np.ndarray:
        """CQA: 模拟量化精度损失 (quantize → dequantize)

        仅在 cqa=True 时调用。跳过 fp32 (无损)。
        """
        if qtype in ("fp32", "float32"):
            return data
        if qtype in ("fp16", "float16"):
            return data.astype(np.float16).astype(np.float32)
        from aidevtools.formats.quantize import simulate_quantize
        return simulate_quantize(data.astype(np.float32), qtype)

    @staticmethod
    def _get_bfp_params(qtype: str):
        """获取 BFP 格式参数 (block_size, mantissa_bits)"""
        _BFP_PARAMS = {
            "bfpp4": (64, 2), "bfpp8": (32, 4), "bfpp16": (16, 8),
        }
        return _BFP_PARAMS.get(qtype)

    def _cqa_calibrate_weight(
        self,
        W_fp32: np.ndarray,
        x_cqa: np.ndarray,
        y_golden: np.ndarray,
        qtype: str,
    ) -> np.ndarray:
        """CQA 权重校准: lstsq + 量化域坐标下降优化

        算法:
            Phase 1 — lstsq 初始化:
                求解 W_opt = lstsq(x_cqa, y_golden) 得到无约束最优权重

            Phase 2 — 量化域逐行优化 (BFP 格式):
                对每行 (每个输出通道):
                a. 遍历候选共享指数 (搜索最优 block exponent)
                b. 在最优指数下, 坐标下降微调每个尾数值
                使 ||x @ w_j - t_j||² 最小

            Phase 3 — 迭代残差补偿:
                量化后计算残差, lstsq 求补偿, 重复

        Args:
            W_fp32: 原始 fp32 权重 [out, in]
            x_cqa: 量化后的输入 [batch, ..., in_features]
            y_golden: fp32 链的理想输出 [batch, ..., out_features]
            qtype: 权重量化类型

        Returns:
            校准并量化后的权重 [out, in]
        """
        in_dim = x_cqa.shape[-1]
        out_dim = y_golden.shape[-1]

        x_flat = x_cqa.reshape(-1, in_dim)
        y_flat = y_golden.reshape(-1, out_dim)

        # Phase 1: lstsq 初始化
        W_cal_T, _, _, _ = np.linalg.lstsq(x_flat, y_flat, rcond=None)
        W = W_cal_T.T.astype(np.float32)

        bfp = self._get_bfp_params(qtype)
        if bfp is None:
            # 非 BFP 格式, 直接量化返回
            return self._cqa_quantize(W, qtype)

        block_size, mantissa_bits = bfp
        max_mant = 2 ** (mantissa_bits - 1) - 1  # bfpp4: 1, bfpp8: 7

        # Phase 2: 逐行量化域优化
        W_q = np.zeros_like(W)
        for j in range(out_dim):
            row = W[j, :]
            target = y_flat[:, j]  # (N,)

            # 将行分块 (一行可能跨多个 block)
            n_elems = len(row)
            n_blocks = (n_elems + block_size - 1) // block_size
            pad_len = n_blocks * block_size - n_elems
            if pad_len > 0:
                row_padded = np.concatenate([row, np.zeros(pad_len, dtype=np.float32)])
                x_padded = np.concatenate(
                    [x_flat, np.zeros((x_flat.shape[0], pad_len), dtype=np.float32)],
                    axis=1,
                )
            else:
                row_padded = row
                x_padded = x_flat

            blocks = row_padded.reshape(n_blocks, block_size)
            best_row_q = np.zeros_like(row_padded)

            # 逐 block 优化
            for b in range(n_blocks):
                blk = blocks[b]
                col_start = b * block_size
                col_end = col_start + block_size
                x_blk = x_padded[:, col_start:col_end]  # (N, block_size)

                # 其他 block 的贡献 (已确定)
                other_contrib = np.zeros(len(target), dtype=np.float32)
                for ob in range(n_blocks):
                    if ob == b:
                        continue
                    if ob < b:
                        # 已优化的 block
                        obs = ob * block_size
                        obe = obs + block_size
                        other_contrib += x_padded[:, obs:obe] @ best_row_q[obs:obe]
                    else:
                        # 未优化的 block — 用 lstsq 初始值的量化版本估算
                        obs = ob * block_size
                        obe = obs + block_size
                        ob_blk = blocks[ob]
                        ob_abs = np.max(np.abs(ob_blk))
                        if ob_abs < 1e-10:
                            continue
                        ob_exp = int(np.floor(np.log2(ob_abs))) + 1
                        ob_scale = 2.0 ** (mantissa_bits - 1 - ob_exp)
                        ob_mant = np.clip(np.round(ob_blk * ob_scale), -max_mant, max_mant)
                        ob_q = ob_mant / ob_scale
                        other_contrib += x_padded[:, obs:obe] @ ob_q

                # 本 block 的目标
                blk_target = target - other_contrib  # (N,)

                # 搜索最优共享指数
                abs_max = np.max(np.abs(blk))
                if abs_max < 1e-10:
                    best_row_q[col_start:col_end] = 0.0
                    continue

                base_exp = int(np.floor(np.log2(abs_max))) + 1
                best_blk_err = float('inf')
                best_blk_q = None
                best_mants = None
                best_exp = base_exp

                for exp_off in range(-2, 3):
                    exp = base_exp + exp_off
                    scale = 2.0 ** (mantissa_bits - 1 - exp)
                    mants = np.clip(np.round(blk * scale), -max_mant, max_mant)
                    blk_q = mants / scale
                    err = np.sum((x_blk @ blk_q - blk_target) ** 2)
                    if err < best_blk_err:
                        best_blk_err = err
                        best_blk_q = blk_q.copy()
                        best_mants = mants.copy()
                        best_exp = exp

                # 坐标下降: 逐元素微调尾数 (仅低精度格式, 高精度跳过)
                if max_mant <= 7:  # bfpp4 (1) 和 bfpp8 (7) 做坐标下降
                    mants = best_mants.copy()
                    scale_inv = 2.0 ** (mantissa_bits - 1 - best_exp)
                    for _sweep in range(2):
                        for k in range(block_size):
                            orig = mants[k]
                            blk_q = mants / scale_inv
                            base_err = np.sum((x_blk @ blk_q - blk_target) ** 2)
                            for trial in range(-max_mant, max_mant + 1):
                                if trial == orig:
                                    continue
                                mants[k] = trial
                                blk_q = mants / scale_inv
                                err = np.sum((x_blk @ blk_q - blk_target) ** 2)
                                if err < base_err:
                                    base_err = err
                                    orig = trial
                                else:
                                    mants[k] = orig
                    best_row_q[col_start:col_end] = mants / scale_inv
                else:
                    # 高精度格式: 仅用 AdaRound (floor vs ceil)
                    mants = best_mants.copy()
                    scale = 2.0 ** (mantissa_bits - 1 - best_exp)
                    blk_raw = blk * scale  # fp32 未取整值
                    for k in range(block_size):
                        floor_v = np.floor(blk_raw[k])
                        ceil_v = np.ceil(blk_raw[k])
                        floor_v = np.clip(floor_v, -max_mant, max_mant)
                        ceil_v = np.clip(ceil_v, -max_mant, max_mant)
                        # 尝试 floor
                        mants[k] = floor_v
                        blk_q = mants / scale
                        err_f = np.sum((x_blk @ blk_q - blk_target) ** 2)
                        # 尝试 ceil
                        mants[k] = ceil_v
                        blk_q = mants / scale
                        err_c = np.sum((x_blk @ blk_q - blk_target) ** 2)
                        mants[k] = floor_v if err_f <= err_c else ceil_v
                    best_row_q[col_start:col_end] = mants / scale

            W_q[j, :] = best_row_q[:n_elems]

        # Phase 3: 迭代残差补偿 (2 轮)
        for _ in range(2):
            residual = y_flat - x_flat @ W_q.T
            dW_T, _, _, _ = np.linalg.lstsq(x_flat, residual, rcond=None)
            W_new = W_q + dW_T.T.astype(np.float32)
            W_q_new = self._cqa_quantize(W_new, qtype)
            if np.mean((x_flat @ W_q_new.T - y_flat) ** 2) < \
               np.mean((x_flat @ W_q.T - y_flat) ** 2):
                W_q = W_q_new

        return W_q

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
        else:
            t = self._gen.randn(shape, name=name, qtype=qtype, role="input")
            array = t.array

        # CQA: 对初始输入做量化
        if self._cqa:
            input_qtype = qtype or self._gen.precision.input_dtype
            array = self._cqa_quantize(array, input_qtype)

        return ModelTensor(shape=array.shape, name=name, golden=array)

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
        from aidevtools.ops.registry import get_op_instance

        op_name = "matmul"
        prefix = self._next_name(op_name)

        op = get_op_instance(op_name)
        if op is None:
            raise ValueError(f"算子 '{op_name}' 未注册")

        golden = op.cpu_golden(x.golden, y.golden)
        if self._cqa:
            golden = self._cqa_quantize(golden, self._gen.precision.output_dtype)
        out = ModelTensor(shape=golden.shape, name=f"{prefix}.output", golden=golden)
        self._outputs.append(out)
        return out

    def add(self, x: ModelTensor, y: ModelTensor, **kwargs) -> ModelTensor:
        """Add"""
        from aidevtools.ops.registry import get_op_instance
        op = get_op_instance("add")
        golden = op.cpu_golden(x.golden, y.golden)
        if self._cqa:
            golden = self._cqa_quantize(golden, self._gen.precision.output_dtype)
        name = self._next_name("add")
        out = ModelTensor(shape=golden.shape, name=f"{name}.output", golden=golden)
        self._outputs.append(out)
        return out

    def mul(self, x: ModelTensor, y: ModelTensor, **kwargs) -> ModelTensor:
        """Mul"""
        from aidevtools.ops.registry import get_op_instance
        op = get_op_instance("mul")
        golden = op.cpu_golden(x.golden, y.golden)
        if self._cqa:
            golden = self._cqa_quantize(golden, self._gen.precision.output_dtype)
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
        """调用算子，自动生成权重

        CQA 模式 (cqa=True) 流程:
            1. 生成 fp32 权重 (同非 CQA)
            2. 用 fp32 权重 + fp32 输入计算理想 golden (参考目标)
            3. 对有权重的线性算子: lstsq 校准权重 → 量化
            4. 用校准量化权重 + 量化输入重新计算
            5. 输出量化后传给下层
        """
        from aidevtools.ops.registry import get_op_meta, get_op_instance

        meta = get_op_meta(op_name)
        if meta is None:
            raise ValueError(f"算子 '{op_name}' 未注册")

        prefix = self._next_name(op_name)
        input_shape = x.shape

        # 根据 auto_gen 生成权重
        context = {"input_shape": input_shape, **kwargs}
        args = [x.golden]  # 第一个参数是输入 (CQA 模式下已是量化后的)

        # 处理非 auto_gen 的 inputs (如 normalized_shape)
        for inp in meta.inputs[1:]:
            if inp not in meta.auto_gen:
                if inp == "normalized_shape":
                    args.append((input_shape[-1],))
                elif inp in kwargs:
                    args.append(kwargs[inp])

        kwargs_call = {}
        pc = self._gen.precision

        # 收集权重信息 (CQA 校准需要)
        weight_params = {}  # {param: (array_fp32, param_qtype, arg_index_or_kwarg)}

        for param, strategy in meta.auto_gen.items():
            if param == meta.inputs[0]:
                continue

            is_weight = param in meta.weight_params
            role = "weight" if is_weight else "input"

            param_qtype = pc.get_dtype(param, is_weight)
            if param_qtype == "fp32":
                param_qtype = self._gen.qtype

            # 生成或加载 fp32 权重
            tensor_name = f"{prefix}.{param}"
            load_key = tensor_name.replace(".", "_")
            if self._loaded and load_key in self._loaded:
                array = self._loaded[load_key]
                shape = array.shape
            else:
                array, shape = self._gen._gen_from_strategy(strategy, context)
            self._gen._add_tensor(tensor_name, array, qtype=param_qtype, role=role)

            array_fp32 = array.copy()

            # 非 CQA: 直接用 fp32 权重
            # CQA 无权重校准: 直接量化
            if self._cqa and not is_weight:
                array = self._cqa_quantize(array, param_qtype)

            context[f"{param}_shape"] = shape
            context[param] = array

            if param in meta.inputs[1:]:
                idx = len(args)
                args.append(array)
                if is_weight:
                    weight_params[param] = (array_fp32, param_qtype, ("arg", idx))
            elif param in meta.optional:
                kwargs_call[param] = array
                if is_weight:
                    weight_params[param] = (array_fp32, param_qtype, ("kwarg", param))

        op = get_op_instance(op_name)

        # 找主权重 (2D 矩阵, 排除 bias 等 1D 参数)
        main_weight = None
        if self._cqa and weight_params and op_name in ("linear",):
            for param, (w_fp32, pqtype, loc) in weight_params.items():
                if w_fp32.ndim == 2:
                    main_weight = (param, w_fp32, pqtype, loc)
                    break

        if self._cqa and main_weight is not None:
            # CQA 权重校准:
            # 1. 先用 fp32 权重计算理想 golden (参考目标)
            args_fp32 = list(args)
            kwargs_fp32 = dict(kwargs_call)
            for param, (w_fp32, _, loc) in weight_params.items():
                if loc[0] == "arg":
                    args_fp32[loc[1]] = w_fp32
                else:
                    kwargs_fp32[loc[1]] = w_fp32
            y_golden = op.cpu_golden(*args_fp32, **kwargs_fp32)

            # 2. lstsq 校准主权重矩阵
            x_cqa = x.golden
            param, w_fp32, pqtype, loc = main_weight
            w_cal = self._cqa_calibrate_weight(w_fp32, x_cqa, y_golden, pqtype)
            if loc[0] == "arg":
                args[loc[1]] = w_cal
            else:
                kwargs_call[loc[1]] = w_cal

            # 3. 其他权重参数 (bias 等) 直接量化
            for p, (wf, pt, lc) in weight_params.items():
                if p == param:
                    continue
                wq = self._cqa_quantize(wf, pt)
                if lc[0] == "arg":
                    args[lc[1]] = wq
                else:
                    kwargs_call[lc[1]] = wq

            # 4. 用校准权重重新计算
            golden = op.cpu_golden(*args, **kwargs_call)
        else:
            # 非 CQA 或无权重/非线性算子: CQA 只做简单量化
            if self._cqa:
                for param, (w_fp32, pqtype, loc) in weight_params.items():
                    w_q = self._cqa_quantize(w_fp32, pqtype)
                    if loc[0] == "arg":
                        args[loc[1]] = w_q
                    else:
                        kwargs_call[loc[1]] = w_q
            golden = op.cpu_golden(*args, **kwargs_call)

        # CQA: 输出量化后再传给下层
        if self._cqa:
            golden = self._cqa_quantize(golden, pc.output_dtype)

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
