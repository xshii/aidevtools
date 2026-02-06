"""
测量数据归档

用于存储 ML 校准的训练数据 (X, Y)

设计:
- X (特征): shape, 算子类型, 融合组合, 硬件配置等
- Y (标签): 实测 cycles, 时延, 带宽利用率等

支持:
- 增量添加样本
- 按条件查询
- 版本管理
- 导出为 ML 框架格式 (numpy, pandas, torch)
"""

import json
import hashlib
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from enum import Enum
import sqlite3


class MeasurementSource(Enum):
    """测量来源"""
    ESL = "esl"              # ESL 仿真
    HARDWARE = "hardware"    # 硬件实测
    PROFILER = "profiler"    # Profiler 工具
    MANUAL = "manual"        # 手动输入
    SYNTHETIC = "synthetic"  # 合成数据


@dataclass
class FeatureVector:
    """
    特征向量 (X)

    包含算子/融合组的所有输入特征
    """
    # 算子基本信息
    op_types: Tuple[str, ...]          # 算子类型序列
    num_ops: int                        # 算子数量

    # Shape 特征
    shapes: Dict[str, int]              # 关键维度 {M, N, K, batch, seq, hidden}
    total_elements: int                 # 总元素数
    total_flops: int                    # 总 FLOPs
    total_bytes: int                    # 总访存量

    # 计算特征
    arithmetic_intensity: float         # 算术强度 (FLOPs/Byte)
    compute_unit: str                   # 计算单元 (cube/vector/mixed)

    # 融合特征
    is_fused: bool                      # 是否融合
    fusion_depth: int                   # 融合深度
    fusion_pattern: str = ""            # 融合模式名称

    # Tile 特征
    tile_sizes: Dict[str, int] = field(default_factory=dict)
    num_tiles: int = 1
    buffer_size: int = 0

    # 硬件配置
    chip_type: str = ""                 # 芯片型号
    frequency_mhz: int = 0              # 频率

    def to_vector(self) -> List[float]:
        """转换为数值向量"""
        vec = [
            float(self.num_ops),
            float(self.total_elements),
            float(self.total_flops),
            float(self.total_bytes),
            self.arithmetic_intensity,
            1.0 if self.is_fused else 0.0,
            float(self.fusion_depth),
            float(self.num_tiles),
            float(self.buffer_size),
        ]

        # Shape 特征
        for key in ["M", "N", "K", "batch", "seq", "hidden"]:
            vec.append(float(self.shapes.get(key, 0)))

        # Tile 特征
        for key in ["M", "N", "K"]:
            vec.append(float(self.tile_sizes.get(key, 0)))

        return vec

    @classmethod
    def feature_names(cls) -> List[str]:
        """特征名列表"""
        return [
            "num_ops", "total_elements", "total_flops", "total_bytes",
            "arithmetic_intensity", "is_fused", "fusion_depth",
            "num_tiles", "buffer_size",
            "shape_M", "shape_N", "shape_K", "shape_batch", "shape_seq", "shape_hidden",
            "tile_M", "tile_N", "tile_K",
        ]

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FeatureVector":
        """从字典创建"""
        return cls(
            op_types=tuple(data.get("op_types", [])),
            num_ops=data.get("num_ops", 0),
            shapes=data.get("shapes", {}),
            total_elements=data.get("total_elements", 0),
            total_flops=data.get("total_flops", 0),
            total_bytes=data.get("total_bytes", 0),
            arithmetic_intensity=data.get("arithmetic_intensity", 0.0),
            compute_unit=data.get("compute_unit", ""),
            is_fused=data.get("is_fused", False),
            fusion_depth=data.get("fusion_depth", 0),
            fusion_pattern=data.get("fusion_pattern", ""),
            tile_sizes=data.get("tile_sizes", {}),
            num_tiles=data.get("num_tiles", 1),
            buffer_size=data.get("buffer_size", 0),
            chip_type=data.get("chip_type", ""),
            frequency_mhz=data.get("frequency_mhz", 0),
        )


@dataclass
class LabelVector:
    """
    标签向量 (Y)

    核心标签: latency_us (时延)
    其他指标可选，用于辅助分析
    """
    # 核心标签 (必须)
    latency_us: float                   # 实测时延 (微秒) - 主要标签

    # 可选: 换算指标
    total_cycles: int = 0               # 总周期数 (可由 latency 换算)
    frequency_mhz: int = 1000           # 频率 (用于换算)

    # 可选: 分解指标 (如有)
    compute_cycles: int = 0             # 计算周期
    memory_cycles: int = 0              # 访存周期
    overhead_cycles: int = 0            # 开销周期

    # 可选: 效率指标 (如有)
    compute_utilization: float = 0.0    # 计算利用率
    memory_bandwidth_util: float = 0.0  # 带宽利用率

    # 可选: 对比指标
    speedup_vs_unfused: float = 1.0     # 相比不融合的加速比
    baseline_latency_us: float = 0.0    # 不融合时的时延 (用于计算 speedup)

    def __post_init__(self):
        """自动换算 cycles"""
        if self.total_cycles == 0 and self.latency_us > 0:
            # latency(us) * frequency(MHz) = cycles
            self.total_cycles = int(self.latency_us * self.frequency_mhz)

    def to_vector(self) -> List[float]:
        """转换为数值向量 (主要用 latency_us)"""
        return [
            self.latency_us,
            float(self.total_cycles),
            self.speedup_vs_unfused,
        ]

    @classmethod
    def label_names(cls) -> List[str]:
        """标签名列表"""
        return ["latency_us", "total_cycles", "speedup_vs_unfused"]

    @property
    def primary_label(self) -> float:
        """主要标签值"""
        return self.latency_us

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "latency_us": self.latency_us,
            "total_cycles": self.total_cycles,
            "frequency_mhz": self.frequency_mhz,
            "compute_cycles": self.compute_cycles,
            "memory_cycles": self.memory_cycles,
            "overhead_cycles": self.overhead_cycles,
            "compute_utilization": self.compute_utilization,
            "memory_bandwidth_util": self.memory_bandwidth_util,
            "speedup_vs_unfused": self.speedup_vs_unfused,
            "baseline_latency_us": self.baseline_latency_us,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LabelVector":
        """从字典创建"""
        return cls(
            latency_us=data.get("latency_us", 0.0),
            total_cycles=data.get("total_cycles", 0),
            frequency_mhz=data.get("frequency_mhz", 1000),
            compute_cycles=data.get("compute_cycles", 0),
            memory_cycles=data.get("memory_cycles", 0),
            overhead_cycles=data.get("overhead_cycles", 0),
            compute_utilization=data.get("compute_utilization", 0.0),
            memory_bandwidth_util=data.get("memory_bandwidth_util", 0.0),
            speedup_vs_unfused=data.get("speedup_vs_unfused", 1.0),
            baseline_latency_us=data.get("baseline_latency_us", 0.0),
        )

    @classmethod
    def from_latency(cls, latency_us: float, frequency_mhz: int = 1000) -> "LabelVector":
        """快捷创建: 只需时延"""
        return cls(latency_us=latency_us, frequency_mhz=frequency_mhz)


@dataclass
class MeasurementSample:
    """
    单个测量样本

    包含 X (特征) 和 Y (标签)
    """
    id: str                             # 唯一标识
    features: FeatureVector             # X
    labels: LabelVector                 # Y

    # 元数据
    source: MeasurementSource           # 数据来源
    timestamp: str                      # 时间戳
    version: str = "1.0"                # 数据版本
    tags: List[str] = field(default_factory=list)  # 标签
    notes: str = ""                     # 备注

    # 预测值 (用于校准后验证)
    predicted_latency_us: Optional[float] = None

    @property
    def error(self) -> Optional[float]:
        """预测误差 (基于 latency_us)"""
        if self.predicted_latency_us is None:
            return None
        actual = self.labels.latency_us
        if actual <= 0:
            return None
        return abs(self.predicted_latency_us - actual) / actual

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "id": self.id,
            "features": self.features.to_dict(),
            "labels": self.labels.to_dict(),
            "source": self.source.value,
            "timestamp": self.timestamp,
            "version": self.version,
            "tags": self.tags,
            "notes": self.notes,
            "predicted_latency_us": self.predicted_latency_us,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MeasurementSample":
        """从字典创建"""
        return cls(
            id=data["id"],
            features=FeatureVector.from_dict(data["features"]),
            labels=LabelVector.from_dict(data["labels"]),
            source=MeasurementSource(data.get("source", "manual")),
            timestamp=data.get("timestamp", ""),
            version=data.get("version", "1.0"),
            tags=data.get("tags", []),
            notes=data.get("notes", ""),
            predicted_latency_us=data.get("predicted_latency_us"),
        )


class MeasurementArchive:
    """
    测量数据归档

    支持:
    - 增量添加样本
    - 按条件查询
    - 导出为 ML 格式
    - 持久化存储 (JSON/SQLite)
    """

    def __init__(self, path: Optional[str] = None):
        """
        Args:
            path: 归档路径 (None 表示内存模式)
        """
        self.path = path
        self.samples: Dict[str, MeasurementSample] = {}
        self._metadata = {
            "version": "1.0",
            "created": datetime.now().isoformat(),
            "updated": datetime.now().isoformat(),
            "description": "",
        }

        if path and Path(path).exists():
            self.load(path)

    # ==================== 样本管理 ====================

    def add(self, sample: MeasurementSample) -> str:
        """添加样本"""
        self.samples[sample.id] = sample
        self._metadata["updated"] = datetime.now().isoformat()
        return sample.id

    def add_measurement(self,
                       features: FeatureVector,
                       labels: LabelVector,
                       source: MeasurementSource = MeasurementSource.MANUAL,
                       tags: Optional[List[str]] = None,
                       notes: str = "") -> str:
        """便捷添加方法"""
        sample_id = self._generate_id(features)
        sample = MeasurementSample(
            id=sample_id,
            features=features,
            labels=labels,
            source=source,
            timestamp=datetime.now().isoformat(),
            tags=tags or [],
            notes=notes,
        )
        return self.add(sample)

    def get(self, sample_id: str) -> Optional[MeasurementSample]:
        """获取样本"""
        return self.samples.get(sample_id)

    def remove(self, sample_id: str) -> bool:
        """删除样本"""
        if sample_id in self.samples:
            del self.samples[sample_id]
            return True
        return False

    def _generate_id(self, features: FeatureVector) -> str:
        """生成样本 ID"""
        content = json.dumps(features.to_dict(), sort_keys=True)
        hash_val = hashlib.md5(content.encode()).hexdigest()[:12]
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        return f"{timestamp}_{hash_val}"

    # ==================== 查询 ====================

    def query(self,
             op_types: Optional[Tuple[str, ...]] = None,
             source: Optional[MeasurementSource] = None,
             tags: Optional[List[str]] = None,
             is_fused: Optional[bool] = None,
             min_elements: Optional[int] = None,
             max_elements: Optional[int] = None,
             chip_type: Optional[str] = None) -> List[MeasurementSample]:
        """
        按条件查询样本

        Args:
            op_types: 算子类型序列
            source: 数据来源
            tags: 包含的标签
            is_fused: 是否融合
            min_elements: 最小元素数
            max_elements: 最大元素数
            chip_type: 芯片类型

        Returns:
            匹配的样本列表
        """
        results = []

        for sample in self.samples.values():
            f = sample.features

            if op_types and f.op_types != op_types:
                continue
            if source and sample.source != source:
                continue
            if tags and not all(t in sample.tags for t in tags):
                continue
            if is_fused is not None and f.is_fused != is_fused:
                continue
            if min_elements and f.total_elements < min_elements:
                continue
            if max_elements and f.total_elements > max_elements:
                continue
            if chip_type and f.chip_type != chip_type:
                continue

            results.append(sample)

        return results

    def list_op_types(self) -> List[Tuple[str, ...]]:
        """列出所有算子类型组合"""
        return list(set(s.features.op_types for s in self.samples.values()))

    def list_tags(self) -> List[str]:
        """列出所有标签"""
        tags = set()
        for s in self.samples.values():
            tags.update(s.tags)
        return list(tags)

    def statistics(self) -> Dict[str, Any]:
        """统计信息"""
        if not self.samples:
            return {"count": 0}

        samples = list(self.samples.values())
        latencies = [s.labels.latency_us for s in samples]

        return {
            "count": len(samples),
            "sources": {
                src.value: sum(1 for s in samples if s.source == src)
                for src in MeasurementSource
            },
            "fused_count": sum(1 for s in samples if s.features.is_fused),
            "unfused_count": sum(1 for s in samples if not s.features.is_fused),
            "latency_us_min": min(latencies),
            "latency_us_max": max(latencies),
            "latency_us_mean": sum(latencies) / len(latencies),
            "op_types_count": len(self.list_op_types()),
        }

    # ==================== 导出 ====================

    def to_numpy(self, label_column: str = "latency_us") -> Tuple:
        """
        导出为 numpy 数组

        Returns:
            (X, y, feature_names, sample_ids)
        """
        import numpy as np

        samples = list(self.samples.values())
        if not samples:
            return np.array([]), np.array([]), [], []

        X = np.array([s.features.to_vector() for s in samples])

        label_idx = LabelVector.label_names().index(label_column)
        y = np.array([s.labels.to_vector()[label_idx] for s in samples])

        feature_names = FeatureVector.feature_names()
        sample_ids = [s.id for s in samples]

        return X, y, feature_names, sample_ids

    def to_pandas(self) -> "pd.DataFrame":
        """导出为 pandas DataFrame"""
        import pandas as pd

        records = []
        for s in self.samples.values():
            record = {
                "id": s.id,
                "source": s.source.value,
                "timestamp": s.timestamp,
            }

            # 特征
            for i, name in enumerate(FeatureVector.feature_names()):
                record[f"x_{name}"] = s.features.to_vector()[i]

            # 标签
            for i, name in enumerate(LabelVector.label_names()):
                record[f"y_{name}"] = s.labels.to_vector()[i]

            # 额外信息
            record["op_types"] = "_".join(s.features.op_types)
            record["is_fused"] = s.features.is_fused

            records.append(record)

        return pd.DataFrame(records)

    def to_torch_dataset(self, label_column: str = "latency_us"):
        """导出为 PyTorch Dataset"""
        import torch
        from torch.utils.data import TensorDataset

        X, y, _, _ = self.to_numpy(label_column)
        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32)

        return TensorDataset(X_tensor, y_tensor)

    # ==================== 持久化 ====================

    def save(self, path: Optional[str] = None) -> None:
        """保存到文件"""
        path = path or self.path
        if not path:
            raise ValueError("No path specified")

        data = {
            "metadata": self._metadata,
            "samples": [s.to_dict() for s in self.samples.values()],
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def load(self, path: str) -> int:
        """从文件加载"""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self._metadata = data.get("metadata", self._metadata)

        count = 0
        for sample_data in data.get("samples", []):
            sample = MeasurementSample.from_dict(sample_data)
            self.samples[sample.id] = sample
            count += 1

        return count

    def export_csv(self, path: str) -> None:
        """导出为 CSV"""
        df = self.to_pandas()
        df.to_csv(path, index=False)

    def import_csv_with_suite(self, path: str,
                             benchmark_suite: "BenchmarkSuite",
                             source: MeasurementSource = MeasurementSource.MANUAL,
                             frequency_mhz: int = 1000) -> int:
        """
        从 CSV 导入 (最简格式，配合 BenchmarkSuite)

        CSV 格式:
            bm_name,latency_us
            bert_ffn_512,125.5
            bert_ffn_512_fused,98.2
            gpt_attention_1024,380.0

        Args:
            path: CSV 文件路径
            benchmark_suite: BenchmarkSuite 实例 (提供 benchmark 定义)
            source: 数据来源
            frequency_mhz: 频率

        Returns:
            导入的样本数
        """
        import pandas as pd

        df = pd.read_csv(path)
        count = 0

        for _, row in df.iterrows():
            bm_name = str(row["bm_name"])
            latency = float(row["latency_us"])

            bm = benchmark_suite.get(bm_name)
            if bm is None:
                print(f"Warning: benchmark '{bm_name}' not found, skipping")
                continue

            features = self._extract_features_from_benchmark(bm, frequency_mhz)
            labels = LabelVector.from_latency(latency, frequency_mhz)

            self.add_measurement(features, labels, source=source, tags=[bm_name])
            count += 1

        return count

    def import_csv(self, path: str,
                  source: MeasurementSource = MeasurementSource.MANUAL,
                  frequency_mhz: int = 1000) -> int:
        """
        从 CSV 导入 (完整格式，无需 BenchmarkSuite)

        CSV 格式要求:
        - 必须列: op_types, latency_us (时延，微秒)
        - 可选列: M, N, K, is_fused, ...

        示例 CSV:
            op_types,M,N,K,is_fused,latency_us
            matmul,512,768,768,false,125.5
            matmul_gelu,512,768,768,true,98.2
            matmul_gelu_matmul,512,768,3072,true,380.0
        """
        import pandas as pd

        df = pd.read_csv(path)
        count = 0

        for _, row in df.iterrows():
            # 解析算子类型
            op_types_str = row.get("op_types", "matmul")
            op_types = tuple(op_types_str.split("_"))

            # 构建特征
            shapes = {}
            for key in ["M", "N", "K", "batch", "seq", "hidden"]:
                if key in row and pd.notna(row[key]):
                    shapes[key] = int(row[key])

            # 计算派生特征
            total_elements = 1
            for v in shapes.values():
                total_elements *= v

            # 估算 FLOPs (简化)
            if "matmul" in op_types:
                M = shapes.get("M", 1)
                N = shapes.get("N", 1)
                K = shapes.get("K", 1)
                total_flops = 2 * M * N * K * op_types.count("matmul")
            else:
                total_flops = total_elements * len(op_types)

            total_bytes = total_elements * 2 * 3  # 估算: fp16, 输入+权重+输出
            ai = total_flops / total_bytes if total_bytes > 0 else 0

            features = FeatureVector(
                op_types=op_types,
                num_ops=len(op_types),
                shapes=shapes,
                total_elements=total_elements,
                total_flops=int(row.get("total_flops", total_flops)),
                total_bytes=int(row.get("total_bytes", total_bytes)),
                arithmetic_intensity=float(row.get("arithmetic_intensity", ai)),
                compute_unit=str(row.get("compute_unit", "auto")),
                is_fused=bool(row.get("is_fused", len(op_types) > 1)),
                fusion_depth=int(row.get("fusion_depth", len(op_types))),
            )

            # 构建标签 - 只需 latency_us
            latency = float(row["latency_us"])
            labels = LabelVector.from_latency(latency, frequency_mhz)

            # 如果有额外标签，填充
            if "speedup" in row and pd.notna(row["speedup"]):
                labels.speedup_vs_unfused = float(row["speedup"])
            if "baseline_latency_us" in row and pd.notna(row["baseline_latency_us"]):
                labels.baseline_latency_us = float(row["baseline_latency_us"])
                if labels.baseline_latency_us > 0:
                    labels.speedup_vs_unfused = labels.baseline_latency_us / latency

            self.add_measurement(features, labels, source=source)
            count += 1

        return count

    def import_from_benchmarks(self,
                              results: Dict[str, float],
                              benchmarks: Dict[str, "Benchmark"],
                              source: MeasurementSource = MeasurementSource.MANUAL,
                              frequency_mhz: int = 1000) -> int:
        """
        从 Benchmark 定义导入实测结果

        Args:
            results: {benchmark_name: latency_us} 映射
            benchmarks: {benchmark_name: Benchmark} 定义
            source: 数据来源
            frequency_mhz: 频率

        示例:
            # 定义 benchmarks
            benchmarks = {
                "bm1": Benchmark("bm1").add_op("mm", "matmul", M=512, N=768, K=768),
                "bm2": Benchmark("bm2").add_op("mm", "matmul", M=512, N=768, K=768)
                                       .add_op("gelu", "gelu", M=512, N=768),
            }

            # 导入实测结果
            results = {"bm1": 125.5, "bm2": 98.2}
            archive.import_from_benchmarks(results, benchmarks)
        """
        count = 0

        for bm_name, latency in results.items():
            if bm_name not in benchmarks:
                print(f"Warning: benchmark '{bm_name}' not found, skipping")
                continue

            bm = benchmarks[bm_name]
            features = self._extract_features_from_benchmark(bm, frequency_mhz)
            labels = LabelVector.from_latency(latency, frequency_mhz)

            self.add_measurement(features, labels, source=source, tags=[bm_name])
            count += 1

        return count

    def import_results(self,
                      results: List[Tuple[str, float]],
                      benchmark_registry: "BenchmarkSuite",
                      source: MeasurementSource = MeasurementSource.MANUAL,
                      frequency_mhz: int = 1000) -> int:
        """
        从 BenchmarkSuite 导入实测结果 (最简接口)

        Args:
            results: [(benchmark_name, latency_us), ...] 列表
            benchmark_registry: BenchmarkSuite 实例
            source: 数据来源
            frequency_mhz: 频率

        示例:
            from aidevtools.theory_optimizer import BenchmarkSuite

            suite = BenchmarkSuite()
            # suite 已包含预定义的 benchmarks

            # 导入实测结果 - 只需 (名称, 时延)
            results = [
                ("bert_ffn_512", 125.5),
                ("bert_ffn_512_fused", 98.2),
                ("gpt_attention_1024", 380.0),
            ]
            archive.import_results(results, suite)
        """
        count = 0

        for bm_name, latency in results:
            bm = benchmark_registry.get(bm_name)
            if bm is None:
                print(f"Warning: benchmark '{bm_name}' not found in suite, skipping")
                continue

            features = self._extract_features_from_benchmark(bm, frequency_mhz)
            labels = LabelVector.from_latency(latency, frequency_mhz)

            self.add_measurement(features, labels, source=source, tags=[bm_name])
            count += 1

        return count

    def _extract_features_from_benchmark(self, benchmark, frequency_mhz: int = 1000) -> FeatureVector:
        """从 Benchmark 提取特征向量"""
        op_types = tuple(op.op_type.value for op in benchmark.ops)

        total_flops = 0
        total_bytes = 0
        shapes = {}

        for op in benchmark.ops:
            profile = op.to_profile()
            total_flops += profile.flops
            total_bytes += profile.input_bytes + profile.output_bytes

            for k, v in op.shapes.items():
                if k in shapes:
                    shapes[k] = max(shapes[k], v)
                else:
                    shapes[k] = v

        total_elements = 1
        for v in shapes.values():
            total_elements *= v

        ai = total_flops / total_bytes if total_bytes > 0 else 0

        # 判断是否融合
        fusion_groups = benchmark.get_fusion_groups() if hasattr(benchmark, 'get_fusion_groups') else []
        is_fused = len(fusion_groups) > 0 or len(benchmark.ops) > 1
        fusion_depth = max(len(g[0]) for g in fusion_groups) if fusion_groups else len(benchmark.ops)

        return FeatureVector(
            op_types=op_types,
            num_ops=len(benchmark.ops),
            shapes=shapes,
            total_elements=total_elements,
            total_flops=total_flops,
            total_bytes=total_bytes,
            arithmetic_intensity=ai,
            compute_unit="mixed",
            is_fused=is_fused,
            fusion_depth=fusion_depth,
            frequency_mhz=frequency_mhz,
        )

    # ==================== 版本管理 ====================

    def create_snapshot(self, name: str, description: str = "") -> str:
        """创建快照"""
        snapshot_path = f"{self.path}.{name}.snapshot"
        snapshot_data = {
            "name": name,
            "description": description,
            "created": datetime.now().isoformat(),
            "parent": self.path,
            "metadata": self._metadata,
            "samples": [s.to_dict() for s in self.samples.values()],
        }

        with open(snapshot_path, "w", encoding="utf-8") as f:
            json.dump(snapshot_data, f, indent=2)

        return snapshot_path

    def __len__(self) -> int:
        return len(self.samples)

    def __iter__(self):
        return iter(self.samples.values())


# ==================== 便捷函数 ====================


def create_sample_from_benchmark(benchmark, tiling_result, measured_latency_us: float,
                                 source: MeasurementSource = MeasurementSource.ESL,
                                 frequency_mhz: int = 1000) -> MeasurementSample:
    """
    从 Benchmark 和 TilingResult 创建样本

    Args:
        benchmark: Benchmark 对象
        tiling_result: TilingResult 对象
        measured_latency_us: 实测时延 (微秒)
        source: 数据来源
        frequency_mhz: 频率 (MHz), 用于换算 cycles

    Returns:
        MeasurementSample
    """
    # 收集特征
    op_types = tuple(op.op_type.value for op in benchmark.ops)

    total_flops = 0
    total_bytes = 0
    shapes = {}

    for op in benchmark.ops:
        profile = op.to_profile()
        total_flops += profile.flops
        total_bytes += profile.input_bytes + profile.output_bytes

        for k, v in op.shapes.items():
            if k in shapes:
                shapes[k] = max(shapes[k], v)
            else:
                shapes[k] = v

    total_elements = 1
    for v in shapes.values():
        total_elements *= v

    ai = total_flops / total_bytes if total_bytes > 0 else 0

    features = FeatureVector(
        op_types=op_types,
        num_ops=len(benchmark.ops),
        shapes=shapes,
        total_elements=total_elements,
        total_flops=total_flops,
        total_bytes=total_bytes,
        arithmetic_intensity=ai,
        compute_unit="mixed",
        is_fused=len(tiling_result.fusion_configs) > 0,
        fusion_depth=max(len(fc.fused_ops) for fc in tiling_result.fusion_configs) if tiling_result.fusion_configs else 1,
        frequency_mhz=frequency_mhz,
    )

    labels = LabelVector(
        latency_us=measured_latency_us,
        frequency_mhz=frequency_mhz,
        compute_utilization=tiling_result.compute_utilization,
        memory_bandwidth_util=tiling_result.memory_utilization,
    )

    sample_id = f"{benchmark.name}_{datetime.now().strftime('%Y%m%d%H%M%S')}"

    return MeasurementSample(
        id=sample_id,
        features=features,
        labels=labels,
        source=source,
        timestamp=datetime.now().isoformat(),
        tags=[benchmark.category] if benchmark.category else [],
    )
