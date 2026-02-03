"""芯片规格定义与加载"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict

import yaml


@dataclass
class ComputeUnitSpec:
    """计算单元规格"""
    fp32_tflops: float = 0.0
    fp16_tflops: float = 0.0
    bf16_tflops: float = 0.0
    int8_tops: float = 0.0
    int4_tops: float = 0.0

    # 混合精度算力
    mixed_precision: Dict[str, float] = field(default_factory=dict)

    # 硬件参数
    shape: tuple = (16, 16, 16)  # Cube/Tensor Core 尺寸
    freq_ghz: float = 1.0

    def get_power(self, dtype: str) -> float:
        """根据 dtype 获取算力 (TFLOPS)"""
        dtype = dtype.lower()
        if dtype in ("fp32", "float32"):
            return self.fp32_tflops
        if dtype in ("fp16", "float16"):
            return self.fp16_tflops
        if dtype in ("bf16", "bfloat16"):
            return self.bf16_tflops or self.fp16_tflops
        if dtype == "int8":
            return self.int8_tops
        if dtype == "int4":
            return self.int4_tops
        return self.fp16_tflops

    def get_mixed_power(self, dtype_a: str, dtype_b: str) -> float:
        """获取混合精度算力"""
        key = f"{dtype_a}_{dtype_b}"
        if key in self.mixed_precision:
            return self.mixed_precision[key]
        # fallback
        return self.get_power(dtype_a)


@dataclass
class VectorUnitSpec:
    """向量单元规格"""
    fp32_gflops: float = 0.0
    fp16_gflops: float = 0.0

    width: int = 256  # 每周期处理元素数
    freq_ghz: float = 1.0

    def get_power(self, dtype: str) -> float:
        """获取算力 (TFLOPS)"""
        dtype = dtype.lower()
        if dtype in ("fp32", "float32"):
            return self.fp32_gflops / 1000
        return self.fp16_gflops / 1000


@dataclass
class MemoryLevelSpec:
    """存储层级规格"""
    capacity_bytes: int = 0
    bandwidth_gbps: float = 0.0
    latency_ns: float = 0.0

    # 不同访存模式效率
    efficiency: Dict[str, float] = field(default_factory=lambda: {
        "sequential": 0.85,
        "strided": 0.50,
        "random": 0.25,
    })

    def get_effective_bandwidth(self, pattern: str = "sequential") -> float:
        """获取有效带宽"""
        eff = self.efficiency.get(pattern, 0.85)
        return self.bandwidth_gbps * eff


@dataclass
class MemorySpec:
    """存储层次规格"""
    l1: MemoryLevelSpec = field(default_factory=MemoryLevelSpec)
    l2: MemoryLevelSpec = field(default_factory=MemoryLevelSpec)
    hbm: MemoryLevelSpec = field(default_factory=MemoryLevelSpec)


@dataclass
class PipelineSpec:
    """流水参数"""
    dma_channels: int = 2
    cube_vector_parallel: bool = True
    prefetch_depth: int = 2
    kernel_launch_us: float = 5.0
    sync_us: float = 2.0


@dataclass
class ChipSpec:
    """完整芯片规格"""
    name: str = ""
    arch: str = "npu"  # "npu" | "gpu"
    version: str = ""

    cube: ComputeUnitSpec = field(default_factory=ComputeUnitSpec)
    vector: VectorUnitSpec = field(default_factory=VectorUnitSpec)
    memory: MemorySpec = field(default_factory=MemorySpec)
    pipeline: PipelineSpec = field(default_factory=PipelineSpec)

    @property
    def cube_ridge_point(self) -> float:
        """Cube Roofline 拐点 (FLOPs/Byte)"""
        if self.memory.hbm.bandwidth_gbps == 0:
            return 0
        return self.cube.fp16_tflops * 1e12 / (self.memory.hbm.bandwidth_gbps * 1e9)

    @property
    def vector_ridge_point(self) -> float:
        """Vector Roofline 拐点 (FLOPs/Byte)"""
        if self.memory.hbm.bandwidth_gbps == 0:
            return 0
        return self.vector.fp16_gflops * 1e9 / (self.memory.hbm.bandwidth_gbps * 1e9)

    def get_compute_power(self, unit: str, dtype: str) -> float:
        """获取计算单元算力"""
        if unit == "cube":
            return self.cube.get_power(dtype)
        return self.vector.get_power(dtype)


# ============================================================
# 芯片配置加载
# ============================================================

# 内置芯片配置
_BUILTIN_CHIPS: Dict[str, ChipSpec] = {}


def _init_builtin_chips():
    """初始化内置芯片配置"""

    # NPU 310
    _BUILTIN_CHIPS["npu_310"] = ChipSpec(
        name="Ascend 310",
        arch="npu",
        version="310",
        cube=ComputeUnitSpec(
            fp32_tflops=4.0,
            fp16_tflops=8.0,
            bf16_tflops=8.0,
            int8_tops=16.0,
            mixed_precision={
                "fp16_fp16": 8.0,
                "fp16_int8": 12.0,
                "fp16_int4": 14.0,
                "int8_int8": 16.0,
            },
            shape=(16, 16, 16),
            freq_ghz=1.0,
        ),
        vector=VectorUnitSpec(
            fp32_gflops=250,
            fp16_gflops=500,
            width=256,
            freq_ghz=1.0,
        ),
        memory=MemorySpec(
            l1=MemoryLevelSpec(
                capacity_bytes=512 * 1024,
                bandwidth_gbps=2048,
                latency_ns=5,
                efficiency={"sequential": 0.95, "strided": 0.70, "random": 0.50},
            ),
            l2=MemoryLevelSpec(
                capacity_bytes=4 * 1024 * 1024,
                bandwidth_gbps=512,
                latency_ns=20,
                efficiency={"sequential": 0.90, "strided": 0.60, "random": 0.40},
            ),
            hbm=MemoryLevelSpec(
                capacity_bytes=8 * 1024 * 1024 * 1024,
                bandwidth_gbps=72,
                latency_ns=100,
                efficiency={"sequential": 0.85, "strided": 0.50, "random": 0.25},
            ),
        ),
        pipeline=PipelineSpec(
            dma_channels=2,
            cube_vector_parallel=True,
            prefetch_depth=2,
            kernel_launch_us=5.0,
            sync_us=2.0,
        ),
    )

    # NPU 910
    _BUILTIN_CHIPS["npu_910"] = ChipSpec(
        name="Ascend 910",
        arch="npu",
        version="910",
        cube=ComputeUnitSpec(
            fp32_tflops=128.0,
            fp16_tflops=256.0,
            bf16_tflops=256.0,
            int8_tops=512.0,
            mixed_precision={
                "fp16_fp16": 256.0,
                "fp16_int8": 384.0,
                "fp16_int4": 448.0,
                "int8_int8": 512.0,
            },
            shape=(16, 16, 16),
            freq_ghz=1.8,
        ),
        vector=VectorUnitSpec(
            fp32_gflops=8000,
            fp16_gflops=16000,
            width=512,
            freq_ghz=1.8,
        ),
        memory=MemorySpec(
            l1=MemoryLevelSpec(
                capacity_bytes=1024 * 1024,
                bandwidth_gbps=4096,
                latency_ns=3,
                efficiency={"sequential": 0.95, "strided": 0.75, "random": 0.55},
            ),
            l2=MemoryLevelSpec(
                capacity_bytes=32 * 1024 * 1024,
                bandwidth_gbps=2048,
                latency_ns=15,
                efficiency={"sequential": 0.92, "strided": 0.65, "random": 0.45},
            ),
            hbm=MemoryLevelSpec(
                capacity_bytes=32 * 1024 * 1024 * 1024,
                bandwidth_gbps=1200,
                latency_ns=80,
                efficiency={"sequential": 0.88, "strided": 0.55, "random": 0.30},
            ),
        ),
        pipeline=PipelineSpec(
            dma_channels=4,
            cube_vector_parallel=True,
            prefetch_depth=3,
            kernel_launch_us=3.0,
            sync_us=1.0,
        ),
    )

    # NVIDIA A100
    _BUILTIN_CHIPS["gpu_a100"] = ChipSpec(
        name="NVIDIA A100",
        arch="gpu",
        version="A100",
        cube=ComputeUnitSpec(
            fp32_tflops=19.5,
            fp16_tflops=312.0,
            bf16_tflops=312.0,
            int8_tops=624.0,
            mixed_precision={
                "fp16_fp16": 312.0,
                "fp16_int8": 468.0,
                "int8_int8": 624.0,
            },
            shape=(8, 8, 4),  # Tensor Core
            freq_ghz=1.41,
        ),
        vector=VectorUnitSpec(
            fp32_gflops=19500,
            fp16_gflops=39000,
            width=32,  # CUDA Core
            freq_ghz=1.41,
        ),
        memory=MemorySpec(
            l1=MemoryLevelSpec(
                capacity_bytes=192 * 1024,  # per SM
                bandwidth_gbps=19000,
                latency_ns=28,
                efficiency={"sequential": 0.95, "strided": 0.80, "random": 0.60},
            ),
            l2=MemoryLevelSpec(
                capacity_bytes=40 * 1024 * 1024,
                bandwidth_gbps=6000,
                latency_ns=200,
                efficiency={"sequential": 0.90, "strided": 0.70, "random": 0.50},
            ),
            hbm=MemoryLevelSpec(
                capacity_bytes=80 * 1024 * 1024 * 1024,
                bandwidth_gbps=2039,
                latency_ns=400,
                efficiency={"sequential": 0.85, "strided": 0.60, "random": 0.35},
            ),
        ),
        pipeline=PipelineSpec(
            dma_channels=8,
            cube_vector_parallel=True,
            prefetch_depth=4,
            kernel_launch_us=2.0,
            sync_us=0.5,
        ),
    )


# 初始化
_init_builtin_chips()


def load_chip_spec(chip_name: str) -> ChipSpec:
    """
    加载芯片规格

    Args:
        chip_name: 芯片名称，如 "npu_310", "npu_910", "gpu_a100"

    Returns:
        ChipSpec
    """
    chip_name = chip_name.lower()

    # 尝试内置配置
    if chip_name in _BUILTIN_CHIPS:
        return _BUILTIN_CHIPS[chip_name]

    # 尝试加载 YAML 文件
    specs_dir = Path(__file__).parent.parent / "specs"

    # 解析芯片名
    if "_" in chip_name:
        arch = chip_name.split("_", 1)[0]
        yaml_path = specs_dir / arch / f"{chip_name}.yaml"
    else:
        yaml_path = specs_dir / f"{chip_name}.yaml"

    if yaml_path.exists():
        return _load_chip_from_yaml(yaml_path)

    raise ValueError(f"Unknown chip: {chip_name}, available: {list_chips()}")


def _load_chip_from_yaml(path: Path) -> ChipSpec:
    """从 YAML 文件加载芯片配置"""
    with open(path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)

    # 构建 ChipSpec
    spec = ChipSpec(
        name=data.get("name", ""),
        arch=data.get("arch", "npu"),
        version=data.get("version", ""),
    )

    # 解析 compute
    if "compute" in data:
        compute = data["compute"]
        if "cube" in compute:
            cube = compute["cube"]
            spec.cube = ComputeUnitSpec(
                fp32_tflops=cube.get("fp32_tflops", 0),
                fp16_tflops=cube.get("fp16_tflops", 0),
                bf16_tflops=cube.get("bf16_tflops", 0),
                int8_tops=cube.get("int8_tops", 0),
                mixed_precision=cube.get("mixed_precision", {}),
                shape=tuple(cube.get("shape", [16, 16, 16])),
                freq_ghz=cube.get("freq_ghz", 1.0),
            )
        if "vector" in compute:
            vector = compute["vector"]
            spec.vector = VectorUnitSpec(
                fp32_gflops=vector.get("fp32_gflops", 0),
                fp16_gflops=vector.get("fp16_gflops", 0),
                width=vector.get("width", 256),
                freq_ghz=vector.get("freq_ghz", 1.0),
            )

    # 解析 memory
    if "memory" in data:
        memory = data["memory"]
        for level in ["l1", "l2", "hbm"]:
            if level in memory:
                lvl = memory[level]
                level_spec = MemoryLevelSpec(
                    capacity_bytes=_parse_size(lvl.get("capacity", 0)),
                    bandwidth_gbps=lvl.get("bandwidth_gbps", 0),
                    latency_ns=lvl.get("latency_ns", 0),
                    efficiency=lvl.get("efficiency", {}),
                )
                setattr(spec.memory, level, level_spec)

    # 解析 pipeline
    if "pipeline" in data:
        pl = data["pipeline"]
        spec.pipeline = PipelineSpec(
            dma_channels=pl.get("dma_channels", 2),
            cube_vector_parallel=pl.get("cube_vector_parallel", True),
            prefetch_depth=pl.get("prefetch_depth", 2),
            kernel_launch_us=pl.get("kernel_launch_us", 5.0),
            sync_us=pl.get("sync_us", 2.0),
        )

    return spec


def _parse_size(value) -> int:
    """解析大小字符串，如 "512KB", "4MB", "8GB" """
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        value = value.upper().strip()
        if value.endswith("KB"):
            return int(float(value[:-2]) * 1024)
        if value.endswith("MB"):
            return int(float(value[:-2]) * 1024 * 1024)
        if value.endswith("GB"):
            return int(float(value[:-2]) * 1024 * 1024 * 1024)
        return int(value)
    return 0


def list_chips() -> list:
    """列出所有可用芯片"""
    return list(_BUILTIN_CHIPS.keys())
