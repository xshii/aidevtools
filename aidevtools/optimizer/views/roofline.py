"""
Roofline 视图

设计模式:
- Template Method 模式: 继承基类渲染流程
- 与 analysis 模块 RooflinePass 集成
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from .base import View, ViewResult, ViewFormat, ViewRegistry
from ..strategy.base import TilingResult, TileConfig


@dataclass
class RooflinePoint:
    """Roofline 图上的点"""
    op_name: str
    arithmetic_intensity: float  # FLOPs / Bytes
    achieved_performance: float  # GFLOPs/s
    peak_performance: float      # GFLOPs/s
    memory_bound: bool
    utilization: float


@ViewRegistry.register("roofline")
class RooflineView(View):
    """
    Roofline 视图

    展示:
    - 算子的算术强度
    - 实际性能 vs 理论峰值
    - 计算/访存瓶颈分析
    """

    def __init__(self,
                 peak_compute: float = 320.0,  # TFLOPs/s
                 peak_bandwidth: float = 2.4e12):  # B/s (HBM)
        """
        Args:
            peak_compute: 峰值计算能力 (TFLOPs/s)
            peak_bandwidth: 峰值带宽 (B/s)
        """
        self.peak_compute = peak_compute * 1e12  # 转换为 FLOPs/s
        self.peak_bandwidth = peak_bandwidth

        # 计算拐点
        self.ridge_point = self.peak_compute / self.peak_bandwidth

    @property
    def name(self) -> str:
        return "roofline"

    @property
    def description(self) -> str:
        return "Roofline 视图：展示计算/访存瓶颈分析"

    def render(self, data: TilingResult,
               format: ViewFormat = ViewFormat.TEXT) -> ViewResult:
        """渲染 Roofline 视图"""
        # 计算各算子的 roofline 点
        points = self._calculate_roofline_points(data)

        if format == ViewFormat.TEXT:
            content = self._render_text(data, points)
        elif format == ViewFormat.JSON:
            content = self._render_json(data, points)
        elif format == ViewFormat.SVG:
            content = self._render_svg(data, points)
        else:
            content = self._render_text(data, points)

        return ViewResult(
            view_type=self.name,
            format=format,
            content=content,
            metadata={
                "peak_compute": self.peak_compute,
                "peak_bandwidth": self.peak_bandwidth,
                "ridge_point": self.ridge_point,
            }
        )

    def _calculate_roofline_points(self, data: TilingResult) -> List[RooflinePoint]:
        """计算 roofline 点"""
        points = []

        # 收集所有配置
        all_configs: Dict[str, TileConfig] = {}
        for fc in data.fusion_configs:
            all_configs.update(fc.tile_configs)
        all_configs.update(data.unfused_configs)

        for op_name, config in all_configs.items():
            op_spec = data.benchmark.ops.get(op_name)
            if not op_spec:
                continue

            # 计算 FLOPs
            flops = self._calculate_flops(config, op_spec)

            # 计算内存访问量
            bytes_accessed = self._calculate_bytes(config)

            # 算术强度
            ai = flops / bytes_accessed if bytes_accessed > 0 else 0

            # 理论峰值性能 (受 roofline 限制)
            if ai < self.ridge_point:
                peak_perf = ai * self.peak_bandwidth
            else:
                peak_perf = self.peak_compute

            # 估算实际性能
            efficiency = max(config.cube_ratio * 0.8, config.vector_ratio * 0.6)
            achieved_perf = peak_perf * efficiency

            points.append(RooflinePoint(
                op_name=op_name,
                arithmetic_intensity=ai,
                achieved_performance=achieved_perf,
                peak_performance=peak_perf,
                memory_bound=ai < self.ridge_point,
                utilization=efficiency
            ))

        return points

    def _calculate_flops(self, config: TileConfig, op_spec) -> float:
        """计算 FLOPs"""
        if op_spec.op_type.value == "matmul":
            M = config.tile_sizes.get("M", 1)
            N = config.tile_sizes.get("N", 1)
            K = config.tile_sizes.get("K", 1)
            return 2.0 * M * N * K  # GEMM: 2*M*N*K

        # 其他算子：简单估算
        elements = 1
        for v in config.tile_sizes.values():
            elements *= v
        return float(elements)

    def _calculate_bytes(self, config: TileConfig) -> float:
        """计算内存访问字节数"""
        total = sum(config.buffer_sizes.values())
        # 考虑 double buffer
        if config.double_buffer:
            total *= 2
        return float(total)

    def _render_text(self, data: TilingResult,
                    points: List[RooflinePoint]) -> str:
        """渲染文本格式"""
        lines = []
        lines.append("=" * 70)
        lines.append("Roofline Analysis")
        lines.append("=" * 70)
        lines.append("")

        # 硬件参数
        lines.append("Hardware Specifications:")
        lines.append("-" * 40)
        lines.append(f"  Peak Compute: {self.peak_compute/1e12:.1f} TFLOPs/s")
        lines.append(f"  Peak Bandwidth: {self.peak_bandwidth/1e12:.2f} TB/s")
        lines.append(f"  Ridge Point: {self.ridge_point:.1f} FLOPs/Byte")
        lines.append("")

        # 各算子分析
        lines.append("Operator Analysis:")
        lines.append("-" * 40)
        lines.append(f"{'Op Name':<15} {'AI':>8} {'Achieved':>12} {'Peak':>12} {'Util':>8} {'Bound':<10}")
        lines.append("-" * 70)

        for p in points:
            bound = "Memory" if p.memory_bound else "Compute"
            lines.append(
                f"{p.op_name:<15} "
                f"{p.arithmetic_intensity:>8.2f} "
                f"{p.achieved_performance/1e12:>10.2f}T "
                f"{p.peak_performance/1e12:>10.2f}T "
                f"{p.utilization:>7.1%} "
                f"{bound:<10}"
            )

        # ASCII Roofline 图
        lines.append("")
        lines.append("Roofline Chart (ASCII):")
        lines.append("-" * 40)
        lines.append(self._render_ascii_roofline(points))

        # 优化建议
        lines.append("")
        lines.append("Optimization Suggestions:")
        lines.append("-" * 40)

        memory_bound = [p for p in points if p.memory_bound]
        compute_bound = [p for p in points if not p.memory_bound]

        if memory_bound:
            lines.append("Memory-bound operators (consider fusion or tiling):")
            for p in memory_bound:
                lines.append(f"  - {p.op_name}: AI={p.arithmetic_intensity:.2f}")

        if compute_bound:
            lines.append("Compute-bound operators (already efficient):")
            for p in compute_bound:
                lines.append(f"  - {p.op_name}: Util={p.utilization:.1%}")

        return "\n".join(lines)

    def _render_ascii_roofline(self, points: List[RooflinePoint]) -> str:
        """渲染 ASCII roofline 图"""
        width = 50
        height = 15

        # 创建画布
        canvas = [[' ' for _ in range(width)] for _ in range(height)]

        # 绘制坐标轴
        for i in range(height):
            canvas[i][0] = '|'
        for j in range(width):
            canvas[height - 1][j] = '-'
        canvas[height - 1][0] = '+'

        # 绘制 roofline
        ridge_x = int(width * 0.5)  # 拐点位置

        # 斜坡部分 (memory bound)
        for x in range(1, ridge_x):
            y = height - 2 - int((height - 3) * x / ridge_x)
            if 0 <= y < height - 1:
                canvas[y][x] = '/'

        # 平台部分 (compute bound)
        for x in range(ridge_x, width - 1):
            canvas[1][x] = '='

        # 绘制算子点
        if points:
            max_ai = max(p.arithmetic_intensity for p in points) * 1.2
            max_ai = max(max_ai, self.ridge_point * 2)

            for p in points:
                x = int((width - 2) * p.arithmetic_intensity / max_ai) + 1
                y_ratio = p.achieved_performance / self.peak_compute
                y = height - 2 - int((height - 3) * y_ratio)

                x = min(max(1, x), width - 2)
                y = min(max(1, y), height - 2)

                canvas[y][x] = 'O'

        # 转换为字符串
        lines = [''.join(row) for row in canvas]

        # 添加标签
        lines.insert(0, "  Perf")
        lines.append("    " + " " * (width // 2 - 5) + "AI (FLOPs/Byte)")

        return "\n  ".join(lines)

    def _render_json(self, data: TilingResult,
                    points: List[RooflinePoint]) -> str:
        """渲染 JSON 格式"""
        import json

        result = {
            "hardware": {
                "peak_compute_tflops": self.peak_compute / 1e12,
                "peak_bandwidth_tbps": self.peak_bandwidth / 1e12,
                "ridge_point": self.ridge_point,
            },
            "operators": [
                {
                    "name": p.op_name,
                    "arithmetic_intensity": p.arithmetic_intensity,
                    "achieved_tflops": p.achieved_performance / 1e12,
                    "peak_tflops": p.peak_performance / 1e12,
                    "utilization": p.utilization,
                    "memory_bound": p.memory_bound,
                }
                for p in points
            ],
            "summary": {
                "memory_bound_count": sum(1 for p in points if p.memory_bound),
                "compute_bound_count": sum(1 for p in points if not p.memory_bound),
                "avg_utilization": sum(p.utilization for p in points) / len(points) if points else 0,
            }
        }

        return json.dumps(result, indent=2)

    def _render_svg(self, data: TilingResult,
                   points: List[RooflinePoint]) -> str:
        """渲染 SVG 格式"""
        width = 600
        height = 400
        margin = 60

        svg = [f'<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">']

        # 背景
        svg.append(f'<rect width="{width}" height="{height}" fill="white"/>')

        # 绘图区域
        plot_width = width - 2 * margin
        plot_height = height - 2 * margin

        # 坐标轴
        svg.append(f'<line x1="{margin}" y1="{height - margin}" '
                  f'x2="{width - margin}" y2="{height - margin}" stroke="black"/>')
        svg.append(f'<line x1="{margin}" y1="{margin}" '
                  f'x2="{margin}" y2="{height - margin}" stroke="black"/>')

        # 计算缩放
        max_ai = self.ridge_point * 4
        max_perf = self.peak_compute

        def x_scale(ai):
            return margin + (ai / max_ai) * plot_width

        def y_scale(perf):
            return height - margin - (perf / max_perf) * plot_height

        # Roofline
        ridge_x = x_scale(self.ridge_point)
        ridge_y = y_scale(self.peak_compute)

        # 斜坡
        svg.append(f'<line x1="{margin}" y1="{height - margin}" '
                  f'x2="{ridge_x}" y2="{ridge_y}" stroke="red" stroke-width="2"/>')

        # 平台
        svg.append(f'<line x1="{ridge_x}" y1="{ridge_y}" '
                  f'x2="{width - margin}" y2="{ridge_y}" stroke="red" stroke-width="2"/>')

        # 算子点
        colors = ["#2196F3", "#4CAF50", "#FF9800", "#9C27B0", "#F44336"]
        for i, p in enumerate(points):
            cx = x_scale(p.arithmetic_intensity)
            cy = y_scale(p.achieved_performance)

            color = colors[i % len(colors)]
            svg.append(f'<circle cx="{cx}" cy="{cy}" r="8" fill="{color}"/>')
            svg.append(f'<text x="{cx + 10}" y="{cy + 4}" font-size="12">{p.op_name}</text>')

        # 标题
        svg.append(f'<text x="{width // 2}" y="30" text-anchor="middle" '
                  f'font-size="16" font-weight="bold">Roofline Model</text>')

        # 轴标签
        svg.append(f'<text x="{width // 2}" y="{height - 10}" '
                  f'text-anchor="middle" font-size="12">Arithmetic Intensity (FLOPs/Byte)</text>')
        svg.append(f'<text x="15" y="{height // 2}" '
                  f'transform="rotate(-90, 15, {height // 2})" '
                  f'text-anchor="middle" font-size="12">Performance (FLOPs/s)</text>')

        svg.append('</svg>')
        return '\n'.join(svg)
