"""
计算视图

设计模式:
- Template Method 模式: 继承基类渲染流程
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .base import View, ViewResult, ViewFormat, ViewRegistry
from ..strategy.base import TilingResult, FusionConfig, TileConfig


@dataclass
class ComputeSlot:
    """计算时隙"""
    cycle_start: int
    cycle_end: int
    op_name: str
    unit: str  # "cube", "vector"
    utilization: float


@ViewRegistry.register("compute")
class ComputeView(View):
    """
    计算视图

    展示:
    - 计算单元使用情况
    - 时序安排
    - 利用率分析
    """

    @property
    def name(self) -> str:
        return "compute"

    @property
    def description(self) -> str:
        return "计算视图：展示计算单元使用和利用率"

    def render(self, data: TilingResult,
               format: ViewFormat = ViewFormat.TEXT) -> ViewResult:
        """渲染计算视图"""
        if format == ViewFormat.TEXT:
            content = self._render_text(data)
        elif format == ViewFormat.JSON:
            content = self._render_json(data)
        elif format == ViewFormat.HTML:
            content = self._render_html(data)
        else:
            content = self._render_text(data)

        return ViewResult(
            view_type=self.name,
            format=format,
            content=content,
            metadata={
                "compute_utilization": data.compute_utilization,
                "estimated_cycles": data.estimated_cycles,
            }
        )

    def _render_text(self, data: TilingResult) -> str:
        """渲染文本格式"""
        lines = []
        lines.append("=" * 60)
        lines.append("Compute View")
        lines.append("=" * 60)
        lines.append("")

        # 计算单元分配
        lines.append("Compute Unit Allocation:")
        lines.append("-" * 40)

        all_configs = {}
        for fc in data.fusion_configs:
            all_configs.update(fc.tile_configs)
        all_configs.update(data.unfused_configs)

        cube_ops = []
        vector_ops = []
        mixed_ops = []

        for name, config in all_configs.items():
            if config.cube_ratio > 0.8:
                cube_ops.append((name, config))
            elif config.vector_ratio > 0.8:
                vector_ops.append((name, config))
            else:
                mixed_ops.append((name, config))

        if cube_ops:
            lines.append("")
            lines.append("Cube Unit:")
            for name, config in cube_ops:
                bar = self._create_bar(config.cube_ratio, 1.0, width=20)
                lines.append(f"  {name}: {bar} {config.cube_ratio:.0%}")

        if vector_ops:
            lines.append("")
            lines.append("Vector Unit:")
            for name, config in vector_ops:
                bar = self._create_bar(config.vector_ratio, 1.0, width=20)
                lines.append(f"  {name}: {bar} {config.vector_ratio:.0%}")

        if mixed_ops:
            lines.append("")
            lines.append("Mixed (Cube + Vector):")
            for name, config in mixed_ops:
                lines.append(f"  {name}:")
                cube_bar = self._create_bar(config.cube_ratio, 1.0, width=15)
                vector_bar = self._create_bar(config.vector_ratio, 1.0, width=15)
                lines.append(f"    Cube:   {cube_bar} {config.cube_ratio:.0%}")
                lines.append(f"    Vector: {vector_bar} {config.vector_ratio:.0%}")

        # 时序图
        lines.append("")
        lines.append("Execution Timeline:")
        lines.append("-" * 40)

        slots = self._build_compute_slots(data)
        timeline = self._render_timeline_text(slots)
        lines.append(timeline)

        # 融合分析
        if data.fusion_configs:
            lines.append("")
            lines.append("Fusion Analysis:")
            lines.append("-" * 40)

            for i, fc in enumerate(data.fusion_configs):
                ops_str = " + ".join(fc.fused_ops)
                lines.append(f"  Group {i + 1}: {ops_str}")
                lines.append(f"    Pipeline depth: {fc.pipeline_depth}")

                # 估算融合收益
                group_cycles = self._estimate_group_cycles(fc, data.benchmark)
                lines.append(f"    Estimated cycles: {self._format_cycles(group_cycles)}")

        # 汇总
        lines.append("")
        lines.append("=" * 60)
        lines.append("Summary")
        lines.append("=" * 60)
        lines.append(f"Total Estimated Cycles: {self._format_cycles(data.estimated_cycles)}")
        lines.append(f"Compute Utilization: {data.compute_utilization:.1%}")

        util_bar = self._create_bar(data.compute_utilization, 1.0, width=30)
        lines.append(f"  {util_bar}")

        return "\n".join(lines)

    def _render_json(self, data: TilingResult) -> str:
        """渲染 JSON 格式"""
        import json

        result = {
            "compute_allocation": {
                "cube_ops": [],
                "vector_ops": [],
                "mixed_ops": [],
            },
            "timeline": [],
            "fusion_groups": [],
            "summary": {
                "total_cycles": data.estimated_cycles,
                "compute_utilization": data.compute_utilization,
            }
        }

        # 分类算子
        all_configs = {}
        for fc in data.fusion_configs:
            all_configs.update(fc.tile_configs)
        all_configs.update(data.unfused_configs)

        for name, config in all_configs.items():
            op_info = {
                "name": name,
                "cube_ratio": config.cube_ratio,
                "vector_ratio": config.vector_ratio,
            }
            if config.cube_ratio > 0.8:
                result["compute_allocation"]["cube_ops"].append(op_info)
            elif config.vector_ratio > 0.8:
                result["compute_allocation"]["vector_ops"].append(op_info)
            else:
                result["compute_allocation"]["mixed_ops"].append(op_info)

        # 时间线
        slots = self._build_compute_slots(data)
        for slot in slots:
            result["timeline"].append({
                "start": slot.cycle_start,
                "end": slot.cycle_end,
                "op": slot.op_name,
                "unit": slot.unit,
                "utilization": slot.utilization,
            })

        # 融合组
        for fc in data.fusion_configs:
            result["fusion_groups"].append({
                "ops": fc.fused_ops,
                "pipeline_depth": fc.pipeline_depth,
            })

        return json.dumps(result, indent=2)

    def _render_html(self, data: TilingResult) -> str:
        """渲染 HTML 格式"""
        html = ['<!DOCTYPE html>']
        html.append('<html><head>')
        html.append('<style>')
        html.append('body { font-family: monospace; }')
        html.append('.bar-container { width: 200px; height: 20px; background: #ddd; display: inline-block; }')
        html.append('.bar-fill { height: 100%; }')
        html.append('.cube { background: #2196F3; }')
        html.append('.vector { background: #4CAF50; }')
        html.append('.timeline { display: flex; flex-wrap: wrap; margin: 10px 0; }')
        html.append('.slot { padding: 5px 10px; margin: 2px; border-radius: 3px; color: white; }')
        html.append('</style>')
        html.append('</head><body>')

        html.append('<h1>Compute View</h1>')

        # 计算单元分配
        html.append('<h2>Compute Unit Allocation</h2>')

        all_configs = {}
        for fc in data.fusion_configs:
            all_configs.update(fc.tile_configs)
        all_configs.update(data.unfused_configs)

        for name, config in all_configs.items():
            html.append(f'<div><strong>{name}</strong></div>')
            html.append('<div style="display: flex; margin-bottom: 10px;">')

            cube_pct = config.cube_ratio * 100
            vector_pct = config.vector_ratio * 100

            html.append(f'<span style="width: 60px;">Cube:</span>')
            html.append(f'<div class="bar-container"><div class="bar-fill cube" style="width:{cube_pct}%"></div></div>')
            html.append(f'<span style="margin-left: 5px;">{config.cube_ratio:.0%}</span>')
            html.append('</div>')

            html.append('<div style="display: flex; margin-bottom: 10px;">')
            html.append(f'<span style="width: 60px;">Vector:</span>')
            html.append(f'<div class="bar-container"><div class="bar-fill vector" style="width:{vector_pct}%"></div></div>')
            html.append(f'<span style="margin-left: 5px;">{config.vector_ratio:.0%}</span>')
            html.append('</div>')

        # 时间线
        html.append('<h2>Execution Timeline</h2>')
        html.append('<div class="timeline">')

        slots = self._build_compute_slots(data)
        for slot in slots:
            color = "#2196F3" if slot.unit == "cube" else "#4CAF50"
            html.append(f'<div class="slot" style="background:{color}">')
            html.append(f'{slot.op_name}<br>{slot.unit}')
            html.append('</div>')
        html.append('</div>')

        # 汇总
        html.append('<h2>Summary</h2>')
        html.append(f'<p>Total Cycles: {self._format_cycles(data.estimated_cycles)}</p>')
        html.append(f'<p>Compute Utilization: {data.compute_utilization:.1%}</p>')

        html.append('</body></html>')
        return '\n'.join(html)

    def _build_compute_slots(self, data: TilingResult) -> List[ComputeSlot]:
        """构建计算时隙"""
        slots = []
        current_cycle = 0

        # 融合组
        for fc in data.fusion_configs:
            for op_name in fc.fused_ops:
                config = fc.tile_configs[op_name]
                op_spec = data.benchmark.ops.get(op_name)

                # 估算周期
                duration = self._estimate_op_duration(config, op_spec)

                # 确定主要计算单元
                unit = "cube" if config.cube_ratio > config.vector_ratio else "vector"
                util = max(config.cube_ratio, config.vector_ratio)

                slots.append(ComputeSlot(
                    cycle_start=current_cycle,
                    cycle_end=current_cycle + duration,
                    op_name=op_name,
                    unit=unit,
                    utilization=util
                ))

                current_cycle += duration

        # 未融合算子
        for op_name, config in data.unfused_configs.items():
            op_spec = data.benchmark.ops.get(op_name)
            duration = self._estimate_op_duration(config, op_spec)

            unit = "cube" if config.cube_ratio > config.vector_ratio else "vector"
            util = max(config.cube_ratio, config.vector_ratio)

            slots.append(ComputeSlot(
                cycle_start=current_cycle,
                cycle_end=current_cycle + duration,
                op_name=op_name,
                unit=unit,
                utilization=util
            ))

            current_cycle += duration

        return slots

    def _estimate_op_duration(self, config: TileConfig, op_spec) -> int:
        """估算算子执行时间"""
        # 简化估算
        compute = 1
        for v in config.tile_sizes.values():
            compute *= v

        efficiency = max(config.cube_ratio * 0.8, config.vector_ratio * 0.6)
        if efficiency > 0:
            return max(1, int(compute / (efficiency * 1000)))
        return max(1, compute // 1000)

    def _estimate_group_cycles(self, fc: FusionConfig, benchmark) -> int:
        """估算融合组周期"""
        total = 0
        for op_name in fc.fused_ops:
            config = fc.tile_configs[op_name]
            op_spec = benchmark.ops.get(op_name) if benchmark else None
            total += self._estimate_op_duration(config, op_spec)

        # 融合收益
        return int(total * 0.8)

    def _render_timeline_text(self, slots: List[ComputeSlot]) -> str:
        """渲染文本时间线"""
        if not slots:
            return "  (empty)"

        lines = []

        # Cube 行
        cube_line = "  Cube:   "
        for slot in slots:
            if slot.unit == "cube":
                cube_line += f"[{slot.op_name}] "
            else:
                cube_line += " " * (len(slot.op_name) + 3)
        lines.append(cube_line)

        # Vector 行
        vector_line = "  Vector: "
        for slot in slots:
            if slot.unit == "vector":
                vector_line += f"[{slot.op_name}] "
            else:
                vector_line += " " * (len(slot.op_name) + 3)
        lines.append(vector_line)

        return "\n".join(lines)
