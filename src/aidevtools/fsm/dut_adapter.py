"""DUT 接口适配层

本模块提供 DUT (Device Under Test) 接口适配，支持：
1. 自研编译器输出的指令序列
2. 功能仿真器接口封装
3. 共享内存/文件接口
4. 回调接口

使用流程:
    Python 模型 → 自研编译器 → DUT 指令 → 功能仿真器 → DUT 输出
                                              ↓
                                    DUTAdapter (本模块)
                                              ↓
                                    GC Hook (比对)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple, Union
import struct
import mmap
import os
import tempfile
import time

import numpy as np

from aidevtools.core.log import logger
from aidevtools.fsm.hooks import OpContext, FSMGoldenHook


class DUTInterfaceType(Enum):
    """DUT 接口类型"""
    CALLBACK = "callback"         # 回调接口
    SHARED_MEMORY = "shared_memory"  # 共享内存
    FILE = "file"                 # 文件接口
    SOCKET = "socket"             # Socket 接口


@dataclass
class DUTConfig:
    """DUT 适配器配置"""
    interface_type: DUTInterfaceType = DUTInterfaceType.CALLBACK

    # 共享内存配置
    shm_name: str = "/fsm_gc_shm"
    shm_size: int = 64 * 1024 * 1024  # 64MB

    # 文件接口配置
    data_dir: str = "/tmp/fsm_gc"
    file_format: str = "bin"      # bin / npy

    # Socket 配置
    host: str = "localhost"
    port: int = 9999

    # 数据格式
    default_dtype: np.dtype = np.float32

    # 超时配置
    timeout_ms: int = 30000       # 30 秒


@dataclass
class DUTInstruction:
    """DUT 指令"""
    op_name: str                  # 算子名称
    op_id: int                    # 算子 ID
    inputs: Dict[str, Any]        # 输入参数（可能是地址或数据）
    shapes: Dict[str, Tuple[int, ...]]  # 形状信息
    output_shape: Tuple[int, ...]  # 输出形状
    extra: Dict[str, Any] = field(default_factory=dict)


class DUTAdapter(ABC):
    """DUT 适配器基类

    将功能仿真器的输出适配到 GC Hook 接口。

    子类需要实现：
    - _read_dut_output(): 从 DUT 读取输出
    - _write_inputs(): 向 DUT 写入输入（可选）

    Example:
        # 使用回调适配器
        adapter = CallbackDUTAdapter(hook=SegmentHook())

        # 仿真器执行时调用
        for insn in instructions:
            output = simulator.execute(insn)
            adapter.on_op_complete(insn, output)

        # 获取比对结果
        results = adapter.get_results()
    """

    def __init__(
        self,
        hook: Optional[FSMGoldenHook] = None,
        config: Optional[DUTConfig] = None,
    ):
        """
        Args:
            hook: GC Hook（用于比对）
            config: 适配器配置
        """
        self.hook = hook
        self.config = config or DUTConfig()
        self._op_counter: Dict[str, int] = {}

    def set_hook(self, hook: FSMGoldenHook):
        """设置 GC Hook"""
        self.hook = hook

    def on_op_complete(
        self,
        insn: DUTInstruction,
        dut_output: np.ndarray,
        inputs: Optional[Dict[str, np.ndarray]] = None,
    ):
        """算子执行完成回调

        Args:
            insn: 当前指令
            dut_output: DUT 输出
            inputs: 输入数据（如果可用）
        """
        if self.hook is None:
            return

        # 构建上下文
        ctx = OpContext(
            op_name=insn.op_name,
            op_id=insn.op_id,
            full_name=f"{insn.op_name}_{insn.op_id}",
            inputs=inputs or {},
            shapes=insn.shapes,
            extra=insn.extra,
        )

        # 调用 Hook
        self.hook.on_op_complete(ctx, dut_output)

    def get_results(self) -> List[Any]:
        """获取比对结果"""
        if self.hook is None:
            return []
        return self.hook.get_results()

    @abstractmethod
    def connect(self):
        """连接到 DUT"""
        pass

    @abstractmethod
    def disconnect(self):
        """断开连接"""
        pass


class CallbackDUTAdapter(DUTAdapter):
    """回调式 DUT 适配器

    最简单的适配方式，仿真器直接调用回调。

    Example:
        adapter = CallbackDUTAdapter(hook=StepHook())

        def execute_op(op, inputs):
            output = compute(op, inputs)
            adapter.on_op_complete(
                DUTInstruction(op_name=op.name, op_id=op.id, ...),
                output,
                inputs,
            )
            return output
    """

    def connect(self):
        logger.info("CallbackDUTAdapter: ready")

    def disconnect(self):
        logger.info("CallbackDUTAdapter: disconnected")


class SharedMemoryDUTAdapter(DUTAdapter):
    """共享内存式 DUT 适配器

    通过共享内存与仿真器交换数据，支持异步比对。

    内存布局:
        [Header: 64 bytes]
        - magic: 4 bytes ("FSGC")
        - version: 4 bytes
        - status: 4 bytes (0=idle, 1=data_ready, 2=comparing, 3=done)
        - op_name_len: 4 bytes
        - op_id: 4 bytes
        - data_offset: 4 bytes
        - data_size: 4 bytes
        - ... (reserved)

        [Op Name: variable]
        [Input Data: variable]
        [Output Data: variable]
    """

    MAGIC = b"FSGC"
    HEADER_SIZE = 64

    STATUS_IDLE = 0
    STATUS_DATA_READY = 1
    STATUS_COMPARING = 2
    STATUS_DONE = 3

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._shm = None
        self._mm = None

    def connect(self):
        """创建/连接共享内存"""
        try:
            # 创建共享内存文件
            shm_path = f"/dev/shm{self.config.shm_name}"
            if not os.path.exists(shm_path):
                with open(shm_path, "wb") as f:
                    f.write(b"\x00" * self.config.shm_size)

            self._shm = open(shm_path, "r+b")
            self._mm = mmap.mmap(self._shm.fileno(), self.config.shm_size)

            # 写入 magic
            self._mm[0:4] = self.MAGIC
            self._mm.flush()

            logger.info(f"SharedMemoryDUTAdapter: connected to {self.config.shm_name}")
        except Exception as e:
            logger.error(f"Failed to create shared memory: {e}")
            raise

    def disconnect(self):
        """断开共享内存"""
        if self._mm:
            self._mm.close()
        if self._shm:
            self._shm.close()
        logger.info("SharedMemoryDUTAdapter: disconnected")

    def poll_and_compare(self, timeout_ms: Optional[int] = None) -> bool:
        """轮询共享内存并执行比对

        Returns:
            True 如果有数据并完成比对，False 如果超时
        """
        if self._mm is None:
            return False

        timeout_ms = timeout_ms or self.config.timeout_ms
        start_time = time.time()

        while True:
            # 读取状态
            status = struct.unpack("I", self._mm[8:12])[0]

            if status == self.STATUS_DATA_READY:
                # 有数据，执行比对
                self._do_compare()
                return True

            # 检查超时
            elapsed = (time.time() - start_time) * 1000
            if elapsed > timeout_ms:
                return False

            time.sleep(0.001)  # 1ms

    def _do_compare(self):
        """从共享内存读取数据并比对"""
        if self._mm is None or self.hook is None:
            return

        # 设置状态为比对中
        self._mm[8:12] = struct.pack("I", self.STATUS_COMPARING)

        try:
            # 读取 header
            op_name_len = struct.unpack("I", self._mm[12:16])[0]
            op_id = struct.unpack("I", self._mm[16:20])[0]
            data_offset = struct.unpack("I", self._mm[20:24])[0]
            data_size = struct.unpack("I", self._mm[24:28])[0]

            # 读取算子名
            op_name = self._mm[self.HEADER_SIZE:self.HEADER_SIZE + op_name_len].decode()

            # 读取输出数据
            data_start = data_offset
            data_end = data_start + data_size
            dut_output = np.frombuffer(
                self._mm[data_start:data_end],
                dtype=self.config.default_dtype,
            ).copy()

            # 构建指令和上下文
            insn = DUTInstruction(
                op_name=op_name,
                op_id=op_id,
                inputs={},
                shapes={},
                output_shape=dut_output.shape,
            )

            self.on_op_complete(insn, dut_output)

        finally:
            # 设置状态为完成
            self._mm[8:12] = struct.pack("I", self.STATUS_DONE)
            self._mm.flush()


class FileDUTAdapter(DUTAdapter):
    """文件式 DUT 适配器

    通过文件与仿真器交换数据，支持离线比对。

    文件格式:
        {data_dir}/
        ├── op_0_matmul_input_a.bin
        ├── op_0_matmul_input_b.bin
        ├── op_0_matmul_output.bin
        ├── op_1_relu_input.bin
        ├── op_1_relu_output.bin
        └── ...
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._data_dir = None

    def connect(self):
        """创建数据目录"""
        self._data_dir = Path(self.config.data_dir)
        self._data_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"FileDUTAdapter: using directory {self._data_dir}")

    def disconnect(self):
        """清理（可选）"""
        logger.info("FileDUTAdapter: disconnected")

    def compare_from_files(self) -> Iterator[Any]:
        """从文件进行离线比对

        Yields:
            每个算子的比对结果
        """
        if self._data_dir is None or self.hook is None:
            return

        # 查找所有输出文件
        output_files = sorted(self._data_dir.glob("*_output.*"))

        for output_file in output_files:
            # 解析文件名: op_{id}_{name}_output.bin
            name_parts = output_file.stem.split("_")
            if len(name_parts) < 3:
                continue

            op_id = int(name_parts[1])
            op_name = "_".join(name_parts[2:-1])  # 处理带下划线的算子名

            # 加载输出数据
            if output_file.suffix == ".npy":
                dut_output = np.load(output_file)
            else:
                dut_output = np.fromfile(output_file, dtype=self.config.default_dtype)

            # 加载输入数据（如果有）
            inputs = self._load_inputs(op_id, op_name)

            # 构建指令
            insn = DUTInstruction(
                op_name=op_name,
                op_id=op_id,
                inputs={},
                shapes={},
                output_shape=dut_output.shape,
            )

            self.on_op_complete(insn, dut_output, inputs)
            yield self.hook.get_results()[-1] if self.hook.get_results() else None

    def _load_inputs(self, op_id: int, op_name: str) -> Dict[str, np.ndarray]:
        """加载输入文件"""
        inputs = {}
        pattern = f"op_{op_id}_{op_name}_input_*"

        for input_file in self._data_dir.glob(pattern + ".*"):
            # 解析参数名
            param_name = input_file.stem.split("_input_")[-1]

            if input_file.suffix == ".npy":
                inputs[param_name] = np.load(input_file)
            else:
                inputs[param_name] = np.fromfile(
                    input_file, dtype=self.config.default_dtype
                )

        return inputs

    def save_for_offline(
        self,
        insn: DUTInstruction,
        dut_output: np.ndarray,
        inputs: Optional[Dict[str, np.ndarray]] = None,
    ):
        """保存数据用于离线比对"""
        if self._data_dir is None:
            return

        prefix = f"op_{insn.op_id}_{insn.op_name}"

        # 保存输出
        output_path = self._data_dir / f"{prefix}_output.{self.config.file_format}"
        if self.config.file_format == "npy":
            np.save(output_path, dut_output)
        else:
            dut_output.tofile(output_path)

        # 保存输入
        if inputs:
            for name, data in inputs.items():
                input_path = self._data_dir / f"{prefix}_input_{name}.{self.config.file_format}"
                if self.config.file_format == "npy":
                    np.save(input_path, data)
                else:
                    data.tofile(input_path)


# ============================================================
# 编译器集成接口
# ============================================================

@dataclass
class CompiledModel:
    """编译后的模型

    由自研编译器生成，包含指令序列和数据布局。
    """
    name: str                              # 模型名称
    instructions: List[DUTInstruction]     # 指令序列
    weights: Dict[str, np.ndarray]         # 权重数据
    input_spec: Dict[str, Tuple[int, ...]]  # 输入规格
    output_spec: Dict[str, Tuple[int, ...]]  # 输出规格
    metadata: Dict[str, Any] = field(default_factory=dict)


class CompilerOutputAdapter:
    """编译器输出适配器

    将自研编译器的输出转换为 DUT 指令序列。

    Example:
        # 编译模型
        compiler = MyCompiler()
        ir = compiler.compile(pytorch_model)

        # 适配
        adapter = CompilerOutputAdapter()
        compiled_model = adapter.from_compiler_ir(ir)

        # 执行并比对
        dut_adapter = CallbackDUTAdapter(hook=SegmentHook())
        for insn in compiled_model.instructions:
            output = simulator.execute(insn)
            dut_adapter.on_op_complete(insn, output)
    """

    def from_compiler_ir(
        self,
        ir: Any,
        name: str = "model",
    ) -> CompiledModel:
        """从编译器 IR 构建 CompiledModel

        Args:
            ir: 编译器输出的中间表示
            name: 模型名称

        Returns:
            CompiledModel
        """
        # 这里是示例实现，需要根据实际编译器 IR 格式调整
        instructions = []
        weights = {}

        # 假设 IR 是一个算子列表
        if hasattr(ir, "__iter__"):
            for i, op in enumerate(ir):
                insn = self._convert_op(op, i)
                instructions.append(insn)

                # 提取权重
                if hasattr(op, "weights"):
                    for w_name, w_data in op.weights.items():
                        weights[f"{op.name}_{i}_{w_name}"] = w_data

        return CompiledModel(
            name=name,
            instructions=instructions,
            weights=weights,
            input_spec=self._extract_input_spec(ir),
            output_spec=self._extract_output_spec(ir),
        )

    def _convert_op(self, op: Any, idx: int) -> DUTInstruction:
        """将编译器算子转换为 DUT 指令"""
        # 示例实现
        return DUTInstruction(
            op_name=getattr(op, "name", "unknown"),
            op_id=idx,
            inputs=getattr(op, "inputs", {}),
            shapes=getattr(op, "shapes", {}),
            output_shape=getattr(op, "output_shape", ()),
            extra=getattr(op, "extra", {}),
        )

    def _extract_input_spec(self, ir: Any) -> Dict[str, Tuple[int, ...]]:
        """提取输入规格"""
        if hasattr(ir, "input_spec"):
            return ir.input_spec
        return {}

    def _extract_output_spec(self, ir: Any) -> Dict[str, Tuple[int, ...]]:
        """提取输出规格"""
        if hasattr(ir, "output_spec"):
            return ir.output_spec
        return {}


# ============================================================
# 便捷函数
# ============================================================

def create_dut_adapter(
    interface_type: Union[str, DUTInterfaceType],
    hook: Optional[FSMGoldenHook] = None,
    **kwargs,
) -> DUTAdapter:
    """创建 DUT 适配器的工厂函数

    Args:
        interface_type: 接口类型
        hook: GC Hook
        **kwargs: 传递给适配器的参数

    Returns:
        对应类型的 DUT 适配器
    """
    if isinstance(interface_type, str):
        interface_type = DUTInterfaceType(interface_type)

    config = DUTConfig(interface_type=interface_type, **kwargs)

    adapter_map = {
        DUTInterfaceType.CALLBACK: CallbackDUTAdapter,
        DUTInterfaceType.SHARED_MEMORY: SharedMemoryDUTAdapter,
        DUTInterfaceType.FILE: FileDUTAdapter,
    }

    adapter_cls = adapter_map.get(interface_type)
    if adapter_cls is None:
        raise ValueError(f"Unsupported interface type: {interface_type}")

    return adapter_cls(hook=hook, config=config)
