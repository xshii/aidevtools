"""
内存流水视图

设计模式:
- Template Method 模式: 继承基类渲染流程
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .base import View, ViewResult, ViewFormat, ViewRegistry
from ..strategy.base import TilingResult, FusionConfig, TileConfig


@dataclass
class MemoryEvent:
    """内存事件"""
    cycle: int
    event_type: str  # "load", "store", "compute"
    buffer_name: str
    size: int
    level: str  # "L1", "L2", "HBM"


@ViewRegistry.register("memory_flow")
class MemoryFlowView(View):
    """
    内存流水视图

    展示:
    - 内存访问时序
    - 数据流动路径
    - buffer 使用情况
    """

    @property
    def name(self) -> str:
        return "memory_flow"

    @property
    def description(self) -> str:
        return "内存流水视图：展示数据流动和 buffer 使用"

    def render(self, data: TilingResult,
               format: ViewFormat = ViewFormat.TEXT) -> ViewResult:
        """渲染内存流水视图"""
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
                "fusion_groups": len(data.fusion_configs),
                "total_memory": data.memory_footprint,
            }
        )

    def _render_text(self, data: TilingResult) -> str:
        """渲染文本格式"""
        lines = []
        lines.append("=" * 60)
        lines.append("Memory Flow View")
        lines.append("=" * 60)
        lines.append("")

        # 融合组
        for i, fc in enumerate(data.fusion_configs):
            lines.append(f"Fusion Group {i + 1}: {' -> '.join(fc.fused_ops)}")
            lines.append("-" * 40)

            # 时间线
            timeline = self._build_timeline(fc)
            lines.append(self._render_timeline_text(timeline))

            # Buffer 使用
            lines.append("")
            lines.append("Buffer Usage:")
            for op_name, config in fc.tile_configs.items():
                total = config.total_buffer_size()
                lines.append(f"  {op_name}:")
                for buf_name, size in config.buffer_sizes.items():
                    bar = self._create_bar(size, total, width=20)
                    lines.append(f"    {buf_name}: {bar} {self._format_bytes(size)}")

            # 共享 buffer
            if fc.shared_buffers:
                lines.append("")
                lines.append("Shared Buffers:")
                for name, size in fc.shared_buffers.items():
                    lines.append(f"  {name}: {self._format_bytes(size)}")

            lines.append("")

        # 未融合算子
        if data.unfused_configs:
            lines.append("Unfused Operators:")
            lines.append("-" * 40)
            for op_name, config in data.unfused_configs.items():
                total = config.total_buffer_size()
                lines.append(f"  {op_name}: {self._format_bytes(total)}")
                for buf_name, size in config.buffer_sizes.items():
                    bar = self._create_bar(size, total, width=20)
                    lines.append(f"    {buf_name}: {bar} {self._format_bytes(size)}")
            lines.append("")

        # 汇总
        lines.append("=" * 60)
        lines.append("Summary")
        lines.append("=" * 60)
        lines.append(f"Total Memory Footprint: {self._format_bytes(data.memory_footprint)}")
        lines.append(f"Memory Utilization: {data.memory_utilization:.1%}")

        return "\n".join(lines)

    def _render_json(self, data: TilingResult) -> str:
        """渲染 JSON 格式"""
        import json

        result = {
            "fusion_groups": [],
            "unfused_ops": {},
            "summary": {
                "total_memory": data.memory_footprint,
                "memory_utilization": data.memory_utilization,
            }
        }

        for fc in data.fusion_configs:
            group = {
                "ops": fc.fused_ops,
                "pipeline_depth": fc.pipeline_depth,
                "tile_configs": {
                    name: config.to_dict()
                    for name, config in fc.tile_configs.items()
                },
                "shared_buffers": fc.shared_buffers,
                "timeline": self._build_timeline_json(fc),
            }
            result["fusion_groups"].append(group)

        for name, config in data.unfused_configs.items():
            result["unfused_ops"][name] = config.to_dict()

        return json.dumps(result, indent=2)

    def _render_html(self, data: TilingResult) -> str:
        """渲染 HTML 格式"""
        html = ['<!DOCTYPE html>']
        html.append('<html><head>')
        html.append('<style>')
        html.append('body { font-family: monospace; }')
        html.append('.timeline { display: flex; margin: 10px 0; }')
        html.append('.event { padding: 5px; margin: 2px; border-radius: 3px; }')
        html.append('.load { background: #4CAF50; color: white; }')
        html.append('.compute { background: #2196F3; color: white; }')
        html.append('.store { background: #FF9800; color: white; }')
        html.append('.buffer-bar { height: 20px; background: #ddd; margin: 5px 0; }')
        html.append('.buffer-fill { height: 100%; background: #4CAF50; }')
        html.append('</style>')
        html.append('</head><body>')

        html.append('<h1>Memory Flow View</h1>')

        for i, fc in enumerate(data.fusion_configs):
            html.append(f'<h2>Fusion Group {i + 1}: {" → ".join(fc.fused_ops)}</h2>')

            # 时间线
            html.append('<div class="timeline">')
            timeline = self._build_timeline(fc)
            for event in timeline:
                html.append(f'<div class="event {event.event_type}">')
                html.append(f'{event.buffer_name}<br>{self._format_bytes(event.size)}')
                html.append('</div>')
            html.append('</div>')

            # Buffer 使用
            html.append('<h3>Buffer Usage</h3>')
            for op_name, config in fc.tile_configs.items():
                total = config.total_buffer_size()
                html.append(f'<h4>{op_name}</h4>')
                for buf_name, size in config.buffer_sizes.items():
                    pct = (size / total * 100) if total > 0 else 0
                    html.append(f'<div>{buf_name}: {self._format_bytes(size)}</div>')
                    html.append(f'<div class="buffer-bar"><div class="buffer-fill" style="width:{pct}%"></div></div>')

        html.append('<h2>Summary</h2>')
        html.append(f'<p>Total Memory: {self._format_bytes(data.memory_footprint)}</p>')
        html.append(f'<p>Utilization: {data.memory_utilization:.1%}</p>')

        html.append('</body></html>')
        return '\n'.join(html)

    def _build_timeline(self, fc: FusionConfig) -> List[MemoryEvent]:
        """构建时间线"""
        events = []
        cycle = 0

        for op_name in fc.fused_ops:
            config = fc.tile_configs[op_name]

            # Load 事件
            for buf_name, size in config.buffer_sizes.items():
                if buf_name in ["A", "B", "input"]:
                    events.append(MemoryEvent(
                        cycle=cycle,
                        event_type="load",
                        buffer_name=f"{op_name}.{buf_name}",
                        size=size,
                        level="L2" if size < 64 * 1024 else "HBM"
                    ))
                    cycle += 1

            # Compute 事件
            events.append(MemoryEvent(
                cycle=cycle,
                event_type="compute",
                buffer_name=op_name,
                size=0,
                level="L1"
            ))
            cycle += 1

            # Store 事件
            for buf_name, size in config.buffer_sizes.items():
                if buf_name in ["C", "output"]:
                    events.append(MemoryEvent(
                        cycle=cycle,
                        event_type="store",
                        buffer_name=f"{op_name}.{buf_name}",
                        size=size,
                        level="L2" if size < 64 * 1024 else "HBM"
                    ))
                    cycle += 1

        return events

    def _build_timeline_json(self, fc: FusionConfig) -> List[Dict]:
        """构建时间线 JSON"""
        events = self._build_timeline(fc)
        return [
            {
                "cycle": e.cycle,
                "type": e.event_type,
                "buffer": e.buffer_name,
                "size": e.size,
                "level": e.level,
            }
            for e in events
        ]

    def _render_timeline_text(self, events: List[MemoryEvent]) -> str:
        """渲染文本时间线"""
        lines = []
        lines.append("Timeline:")

        # 符号映射
        symbols = {"load": "▼", "compute": "●", "store": "▲"}
        colors = {"load": "L", "compute": "C", "store": "S"}

        # 简化视图
        timeline = ""
        for event in events:
            timeline += f"[{colors[event.event_type]}:{event.buffer_name.split('.')[-1]}] "

        lines.append(f"  {timeline}")

        # 图例
        lines.append("")
        lines.append("  Legend: L=Load, C=Compute, S=Store")

        return "\n".join(lines)
