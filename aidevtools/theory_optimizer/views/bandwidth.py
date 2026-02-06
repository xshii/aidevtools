"""
带宽流水视图

设计模式:
- Template Method 模式: 继承基类渲染流程
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .base import View, ViewResult, ViewFormat, ViewRegistry
from ..strategy.base import TilingResult, FusionConfig, TileConfig


@dataclass
class BandwidthSlot:
    """带宽时隙"""
    cycle_start: int
    cycle_end: int
    channel: str       # "L1_L2", "L2_HBM"
    direction: str     # "load", "store"
    op_name: str
    bytes: int
    bandwidth_util: float  # 0.0 - 1.0


@dataclass
class ChannelStats:
    """通道统计"""
    channel: str
    total_bytes: int
    total_cycles: int
    avg_bandwidth: float
    peak_bandwidth: float
    utilization: float


@ViewRegistry.register("bandwidth_pipeline")
class BandwidthPipelineView(View):
    """
    带宽流水视图

    展示:
    - 各层级内存的带宽使用
    - 数据传输时序
    - 带宽瓶颈分析
    """

    def __init__(self,
                 l1_bandwidth: float = 256e9,    # 256 GB/s
                 l2_bandwidth: float = 128e9,    # 128 GB/s
                 hbm_bandwidth: float = 2.4e12): # 2.4 TB/s
        """
        Args:
            l1_bandwidth: L1 cache 带宽 (B/s)
            l2_bandwidth: L2 cache 带宽 (B/s)
            hbm_bandwidth: HBM 带宽 (B/s)
        """
        self.bandwidths = {
            "L1_L2": l2_bandwidth,
            "L2_HBM": hbm_bandwidth,
        }

    @property
    def name(self) -> str:
        return "bandwidth_pipeline"

    @property
    def description(self) -> str:
        return "带宽流水视图：展示内存层级带宽使用"

    def render(self, data: TilingResult,
               format: ViewFormat = ViewFormat.TEXT) -> ViewResult:
        """渲染带宽流水视图"""
        # 构建带宽时隙
        slots = self._build_bandwidth_slots(data)

        # 计算统计信息
        stats = self._calculate_stats(slots)

        if format == ViewFormat.TEXT:
            content = self._render_text(data, slots, stats)
        elif format == ViewFormat.JSON:
            content = self._render_json(data, slots, stats)
        elif format == ViewFormat.HTML:
            content = self._render_html(data, slots, stats)
        else:
            content = self._render_text(data, slots, stats)

        return ViewResult(
            view_type=self.name,
            format=format,
            content=content,
            metadata={
                "channels": list(self.bandwidths.keys()),
                "total_transfer": sum(s.bytes for s in slots),
            }
        )

    def _build_bandwidth_slots(self, data: TilingResult) -> List[BandwidthSlot]:
        """构建带宽时隙"""
        slots = []
        current_cycle = 0

        # 处理融合组
        for fc in data.fusion_configs:
            group_slots = self._build_fusion_group_slots(fc, current_cycle)
            slots.extend(group_slots)

            if group_slots:
                current_cycle = max(s.cycle_end for s in group_slots)

        # 处理未融合算子
        for op_name, config in data.unfused_configs.items():
            op_slots = self._build_op_slots(op_name, config, current_cycle)
            slots.extend(op_slots)

            if op_slots:
                current_cycle = max(s.cycle_end for s in op_slots)

        return slots

    def _build_fusion_group_slots(self, fc: FusionConfig,
                                  start_cycle: int) -> List[BandwidthSlot]:
        """构建融合组的带宽时隙"""
        slots = []
        cycle = start_cycle

        # 流水线执行：多个算子的数据传输可以重叠
        pipeline_depth = fc.pipeline_depth

        for stage in range(len(fc.fused_ops) + pipeline_depth - 1):
            stage_slots = []

            for i, op_name in enumerate(fc.fused_ops):
                # 检查该算子在当前 stage 是否活跃
                op_stage = stage - i

                if 0 <= op_stage < pipeline_depth:
                    config = fc.tile_configs[op_name]

                    if op_stage == 0:
                        # Load 阶段
                        load_bytes = sum(
                            size for name, size in config.buffer_sizes.items()
                            if name in ["A", "B", "input"]
                        )
                        if load_bytes > 0:
                            # L2 -> L1
                            stage_slots.append(BandwidthSlot(
                                cycle_start=cycle,
                                cycle_end=cycle + 1,
                                channel="L1_L2",
                                direction="load",
                                op_name=op_name,
                                bytes=load_bytes,
                                bandwidth_util=0.7
                            ))

                            # HBM -> L2 (如果需要)
                            if load_bytes > 64 * 1024:
                                stage_slots.append(BandwidthSlot(
                                    cycle_start=cycle,
                                    cycle_end=cycle + 1,
                                    channel="L2_HBM",
                                    direction="load",
                                    op_name=op_name,
                                    bytes=load_bytes,
                                    bandwidth_util=0.6
                                ))

                    elif op_stage == pipeline_depth - 1:
                        # Store 阶段
                        store_bytes = sum(
                            size for name, size in config.buffer_sizes.items()
                            if name in ["C", "output"]
                        )
                        if store_bytes > 0:
                            stage_slots.append(BandwidthSlot(
                                cycle_start=cycle,
                                cycle_end=cycle + 1,
                                channel="L1_L2",
                                direction="store",
                                op_name=op_name,
                                bytes=store_bytes,
                                bandwidth_util=0.6
                            ))

            slots.extend(stage_slots)
            cycle += 1

        return slots

    def _build_op_slots(self, op_name: str, config: TileConfig,
                       start_cycle: int) -> List[BandwidthSlot]:
        """构建单个算子的带宽时隙"""
        slots = []
        cycle = start_cycle

        # Load
        load_bytes = sum(
            size for name, size in config.buffer_sizes.items()
            if name in ["A", "B", "input"]
        )
        if load_bytes > 0:
            slots.append(BandwidthSlot(
                cycle_start=cycle,
                cycle_end=cycle + 1,
                channel="L1_L2",
                direction="load",
                op_name=op_name,
                bytes=load_bytes,
                bandwidth_util=0.5
            ))
            cycle += 1

        # Compute (no bandwidth)
        cycle += 1

        # Store
        store_bytes = sum(
            size for name, size in config.buffer_sizes.items()
            if name in ["C", "output"]
        )
        if store_bytes > 0:
            slots.append(BandwidthSlot(
                cycle_start=cycle,
                cycle_end=cycle + 1,
                channel="L1_L2",
                direction="store",
                op_name=op_name,
                bytes=store_bytes,
                bandwidth_util=0.5
            ))

        return slots

    def _calculate_stats(self, slots: List[BandwidthSlot]) -> Dict[str, ChannelStats]:
        """计算各通道统计"""
        stats = {}

        for channel, peak_bw in self.bandwidths.items():
            channel_slots = [s for s in slots if s.channel == channel]

            if not channel_slots:
                stats[channel] = ChannelStats(
                    channel=channel,
                    total_bytes=0,
                    total_cycles=0,
                    avg_bandwidth=0,
                    peak_bandwidth=peak_bw,
                    utilization=0
                )
                continue

            total_bytes = sum(s.bytes for s in channel_slots)
            total_cycles = sum(s.cycle_end - s.cycle_start for s in channel_slots)

            avg_util = sum(s.bandwidth_util for s in channel_slots) / len(channel_slots)

            stats[channel] = ChannelStats(
                channel=channel,
                total_bytes=total_bytes,
                total_cycles=total_cycles,
                avg_bandwidth=peak_bw * avg_util,
                peak_bandwidth=peak_bw,
                utilization=avg_util
            )

        return stats

    def _render_text(self, data: TilingResult,
                    slots: List[BandwidthSlot],
                    stats: Dict[str, ChannelStats]) -> str:
        """渲染文本格式"""
        lines = []
        lines.append("=" * 70)
        lines.append("Bandwidth Pipeline View")
        lines.append("=" * 70)
        lines.append("")

        # 硬件参数
        lines.append("Memory Hierarchy Bandwidth:")
        lines.append("-" * 40)
        for channel, bw in self.bandwidths.items():
            lines.append(f"  {channel}: {bw/1e9:.0f} GB/s")
        lines.append("")

        # 通道统计
        lines.append("Channel Statistics:")
        lines.append("-" * 40)
        lines.append(f"{'Channel':<12} {'Total':<12} {'Cycles':<8} {'Avg BW':<12} {'Util':<8}")
        lines.append("-" * 55)

        for channel, stat in stats.items():
            lines.append(
                f"{channel:<12} "
                f"{self._format_bytes(stat.total_bytes):<12} "
                f"{stat.total_cycles:<8} "
                f"{stat.avg_bandwidth/1e9:.1f} GB/s   "
                f"{stat.utilization:.1%}"
            )

        # 时间线
        lines.append("")
        lines.append("Bandwidth Timeline:")
        lines.append("-" * 40)

        # 按通道分组
        for channel in self.bandwidths:
            channel_slots = [s for s in slots if s.channel == channel]
            if not channel_slots:
                continue

            lines.append(f"\n{channel}:")
            timeline = self._render_timeline_text(channel_slots)
            lines.append(timeline)

        # 瓶颈分析
        lines.append("")
        lines.append("Bottleneck Analysis:")
        lines.append("-" * 40)

        bottleneck_channel = max(stats.values(), key=lambda s: s.utilization)
        lines.append(f"  Primary bottleneck: {bottleneck_channel.channel}")
        lines.append(f"  Utilization: {bottleneck_channel.utilization:.1%}")

        # 优化建议
        lines.append("")
        lines.append("Optimization Suggestions:")
        for channel, stat in stats.items():
            if stat.utilization > 0.8:
                lines.append(f"  - {channel}: Consider reducing data transfer")
            elif stat.utilization < 0.3:
                lines.append(f"  - {channel}: Underutilized, can increase tile size")

        return "\n".join(lines)

    def _render_timeline_text(self, slots: List[BandwidthSlot]) -> str:
        """渲染文本时间线"""
        if not slots:
            return "  (no activity)"

        lines = []
        lines.append("  Load:  " + " ".join(
            f"[{s.op_name}:{self._format_bytes(s.bytes)}]"
            for s in slots if s.direction == "load"
        ))
        lines.append("  Store: " + " ".join(
            f"[{s.op_name}:{self._format_bytes(s.bytes)}]"
            for s in slots if s.direction == "store"
        ))

        return "\n".join(lines)

    def _render_json(self, data: TilingResult,
                    slots: List[BandwidthSlot],
                    stats: Dict[str, ChannelStats]) -> str:
        """渲染 JSON 格式"""
        import json

        result = {
            "hardware": {
                channel: bw / 1e9  # GB/s
                for channel, bw in self.bandwidths.items()
            },
            "slots": [
                {
                    "cycle_start": s.cycle_start,
                    "cycle_end": s.cycle_end,
                    "channel": s.channel,
                    "direction": s.direction,
                    "op": s.op_name,
                    "bytes": s.bytes,
                    "utilization": s.bandwidth_util,
                }
                for s in slots
            ],
            "stats": {
                channel: {
                    "total_bytes": stat.total_bytes,
                    "total_cycles": stat.total_cycles,
                    "avg_bandwidth_gbps": stat.avg_bandwidth / 1e9,
                    "peak_bandwidth_gbps": stat.peak_bandwidth / 1e9,
                    "utilization": stat.utilization,
                }
                for channel, stat in stats.items()
            },
            "bottleneck": max(stats.keys(), key=lambda c: stats[c].utilization)
            if stats else None
        }

        return json.dumps(result, indent=2)

    def _render_html(self, data: TilingResult,
                    slots: List[BandwidthSlot],
                    stats: Dict[str, ChannelStats]) -> str:
        """渲染 HTML 格式"""
        html = ['<!DOCTYPE html>']
        html.append('<html><head>')
        html.append('<style>')
        html.append('body { font-family: monospace; }')
        html.append('table { border-collapse: collapse; margin: 10px 0; }')
        html.append('th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }')
        html.append('th { background: #f4f4f4; }')
        html.append('.bar { height: 20px; background: #ddd; }')
        html.append('.bar-fill { height: 100%; }')
        html.append('.load { background: #4CAF50; }')
        html.append('.store { background: #FF9800; }')
        html.append('.timeline { display: flex; flex-wrap: wrap; margin: 10px 0; }')
        html.append('.slot { padding: 5px; margin: 2px; border-radius: 3px; color: white; font-size: 11px; }')
        html.append('</style>')
        html.append('</head><body>')

        html.append('<h1>Bandwidth Pipeline View</h1>')

        # 通道统计表
        html.append('<h2>Channel Statistics</h2>')
        html.append('<table>')
        html.append('<tr><th>Channel</th><th>Total Bytes</th><th>Avg BW</th><th>Utilization</th></tr>')

        for channel, stat in stats.items():
            util_pct = stat.utilization * 100
            html.append('<tr>')
            html.append(f'<td>{channel}</td>')
            html.append(f'<td>{self._format_bytes(stat.total_bytes)}</td>')
            html.append(f'<td>{stat.avg_bandwidth/1e9:.1f} GB/s</td>')
            html.append(f'<td>')
            html.append(f'<div class="bar" style="width:200px">')
            html.append(f'<div class="bar-fill load" style="width:{util_pct}%"></div>')
            html.append(f'</div> {stat.utilization:.1%}</td>')
            html.append('</tr>')

        html.append('</table>')

        # 时间线
        html.append('<h2>Timeline</h2>')

        for channel in self.bandwidths:
            channel_slots = [s for s in slots if s.channel == channel]
            if not channel_slots:
                continue

            html.append(f'<h3>{channel}</h3>')
            html.append('<div class="timeline">')

            for s in channel_slots:
                color = "#4CAF50" if s.direction == "load" else "#FF9800"
                html.append(f'<div class="slot" style="background:{color}">')
                html.append(f'{s.op_name}<br>{s.direction}<br>{self._format_bytes(s.bytes)}')
                html.append('</div>')

            html.append('</div>')

        html.append('</body></html>')
        return '\n'.join(html)
