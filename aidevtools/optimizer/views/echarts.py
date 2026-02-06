"""
ECharts 转换插件

将各类视图数据转换为 ECharts 配置
"""

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
from enum import Enum


class ChartType(Enum):
    """图表类型"""
    LINE = "line"
    BAR = "bar"
    PIE = "pie"
    SCATTER = "scatter"
    HEATMAP = "heatmap"
    RADAR = "radar"
    GAUGE = "gauge"
    SANKEY = "sankey"
    TREEMAP = "treemap"


@dataclass
class EChartsOption:
    """ECharts 配置"""
    title: Dict[str, Any] = field(default_factory=dict)
    tooltip: Dict[str, Any] = field(default_factory=dict)
    legend: Dict[str, Any] = field(default_factory=dict)
    xAxis: Optional[Dict[str, Any]] = None
    yAxis: Optional[Dict[str, Any]] = None
    series: List[Dict[str, Any]] = field(default_factory=list)
    grid: Optional[Dict[str, Any]] = None
    toolbox: Optional[Dict[str, Any]] = None
    dataZoom: Optional[List[Dict[str, Any]]] = None

    # 额外配置
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        result = {}

        if self.title:
            result["title"] = self.title
        if self.tooltip:
            result["tooltip"] = self.tooltip
        if self.legend:
            result["legend"] = self.legend
        if self.xAxis:
            result["xAxis"] = self.xAxis
        if self.yAxis:
            result["yAxis"] = self.yAxis
        if self.series:
            result["series"] = self.series
        if self.grid:
            result["grid"] = self.grid
        if self.toolbox:
            result["toolbox"] = self.toolbox
        if self.dataZoom:
            result["dataZoom"] = self.dataZoom

        result.update(self.extra)
        return result

    def to_json(self, indent: int = 2) -> str:
        """转换为 JSON 字符串"""
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    def to_html(self, width: str = "100%", height: str = "400px",
                div_id: str = "chart") -> str:
        """生成完整的 HTML 页面"""
        return f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>{self.title.get('text', 'Chart')}</title>
    <script src="https://cdn.jsdelivr.net/npm/echarts@5/dist/echarts.min.js"></script>
</head>
<body>
    <div id="{div_id}" style="width: {width}; height: {height};"></div>
    <script>
        var chart = echarts.init(document.getElementById('{div_id}'));
        var option = {self.to_json()};
        chart.setOption(option);
        window.addEventListener('resize', function() {{ chart.resize(); }});
    </script>
</body>
</html>"""

    def to_script(self, div_id: str = "chart") -> str:
        """生成嵌入式 JavaScript 代码"""
        return f"""var chart = echarts.init(document.getElementById('{div_id}'));
var option = {self.to_json()};
chart.setOption(option);"""


class EChartsConverter:
    """
    ECharts 转换器

    将各类数据转换为 ECharts 配置
    """

    # 默认配色方案
    DEFAULT_COLORS = [
        "#5470c6", "#91cc75", "#fac858", "#ee6666",
        "#73c0de", "#3ba272", "#fc8452", "#9a60b4",
    ]

    @classmethod
    def line_chart(cls,
                   x_data: List[Any],
                   series_data: Dict[str, List[float]],
                   title: str = "",
                   x_label: str = "",
                   y_label: str = "",
                   smooth: bool = True,
                   area: bool = False) -> EChartsOption:
        """
        创建折线图

        Args:
            x_data: X 轴数据
            series_data: {系列名: 数据列表}
            title: 标题
            x_label: X 轴标签
            y_label: Y 轴标签
            smooth: 是否平滑曲线
            area: 是否填充区域
        """
        series = []
        for name, data in series_data.items():
            s = {
                "name": name,
                "type": "line",
                "data": data,
                "smooth": smooth,
            }
            if area:
                s["areaStyle"] = {}
            series.append(s)

        return EChartsOption(
            title={"text": title} if title else {},
            tooltip={"trigger": "axis"},
            legend={"data": list(series_data.keys())},
            xAxis={
                "type": "category",
                "data": x_data,
                "name": x_label,
            },
            yAxis={
                "type": "value",
                "name": y_label,
            },
            series=series,
            grid={"left": "3%", "right": "4%", "bottom": "3%", "containLabel": True},
        )

    @classmethod
    def bar_chart(cls,
                  x_data: List[str],
                  series_data: Dict[str, List[float]],
                  title: str = "",
                  x_label: str = "",
                  y_label: str = "",
                  stack: bool = False,
                  horizontal: bool = False) -> EChartsOption:
        """
        创建柱状图

        Args:
            x_data: X 轴数据
            series_data: {系列名: 数据列表}
            title: 标题
            stack: 是否堆叠
            horizontal: 是否水平显示
        """
        series = []
        for name, data in series_data.items():
            s = {
                "name": name,
                "type": "bar",
                "data": data,
            }
            if stack:
                s["stack"] = "total"
            series.append(s)

        x_axis = {
            "type": "category",
            "data": x_data,
            "name": x_label,
        }
        y_axis = {
            "type": "value",
            "name": y_label,
        }

        if horizontal:
            x_axis, y_axis = y_axis, x_axis

        return EChartsOption(
            title={"text": title} if title else {},
            tooltip={"trigger": "axis", "axisPointer": {"type": "shadow"}},
            legend={"data": list(series_data.keys())},
            xAxis=x_axis,
            yAxis=y_axis,
            series=series,
            grid={"left": "3%", "right": "4%", "bottom": "3%", "containLabel": True},
        )

    @classmethod
    def pie_chart(cls,
                  data: Dict[str, float],
                  title: str = "",
                  radius: Union[str, List[str]] = "50%",
                  rose: bool = False) -> EChartsOption:
        """
        创建饼图

        Args:
            data: {名称: 值}
            title: 标题
            radius: 半径，如 "50%" 或 ["40%", "70%"] (环形)
            rose: 是否南丁格尔玫瑰图
        """
        pie_data = [{"name": k, "value": v} for k, v in data.items()]

        series_config = {
            "name": title or "分布",
            "type": "pie",
            "radius": radius,
            "data": pie_data,
            "emphasis": {
                "itemStyle": {
                    "shadowBlur": 10,
                    "shadowOffsetX": 0,
                    "shadowColor": "rgba(0, 0, 0, 0.5)",
                }
            },
        }

        if rose:
            series_config["roseType"] = "area"

        return EChartsOption(
            title={"text": title, "left": "center"} if title else {},
            tooltip={"trigger": "item", "formatter": "{a} <br/>{b}: {c} ({d}%)"},
            legend={"orient": "vertical", "left": "left"},
            series=[series_config],
        )

    @classmethod
    def scatter_chart(cls,
                      data: List[List[float]],
                      title: str = "",
                      x_label: str = "",
                      y_label: str = "",
                      symbol_size: int = 10) -> EChartsOption:
        """
        创建散点图

        Args:
            data: [[x, y], ...] 或 [[x, y, size], ...]
            title: 标题
            x_label: X 轴标签
            y_label: Y 轴标签
            symbol_size: 点大小
        """
        return EChartsOption(
            title={"text": title} if title else {},
            tooltip={"trigger": "item"},
            xAxis={"type": "value", "name": x_label},
            yAxis={"type": "value", "name": y_label},
            series=[{
                "type": "scatter",
                "data": data,
                "symbolSize": symbol_size,
            }],
            grid={"left": "3%", "right": "4%", "bottom": "3%", "containLabel": True},
        )

    @classmethod
    def roofline_chart(cls,
                       peak_compute: float,
                       peak_bandwidth: float,
                       points: List[Dict[str, Any]],
                       title: str = "Roofline Model") -> EChartsOption:
        """
        创建 Roofline 图

        Args:
            peak_compute: 峰值算力 (GFLOPS)
            peak_bandwidth: 峰值带宽 (GB/s)
            points: [{"name": str, "ai": float, "perf": float}, ...]
        """
        # 计算拐点
        ridge_point = peak_compute / peak_bandwidth

        # Roofline 曲线数据
        ai_range = [0.01, 0.1, 1, 10, 100, 1000]
        roofline_data = []
        for ai in ai_range:
            perf = min(peak_compute, ai * peak_bandwidth)
            roofline_data.append([ai, perf])

        # 散点数据
        scatter_data = [[p["ai"], p["perf"]] for p in points]
        point_names = [p["name"] for p in points]

        return EChartsOption(
            title={"text": title},
            tooltip={
                "trigger": "item",
                "formatter": lambda p: f"{p['seriesName']}: AI={p['data'][0]:.2f}, Perf={p['data'][1]:.1f} GFLOPS"
            },
            xAxis={
                "type": "log",
                "name": "Arithmetic Intensity (FLOP/Byte)",
                "min": 0.01,
                "max": 1000,
            },
            yAxis={
                "type": "log",
                "name": "Performance (GFLOPS)",
                "min": 1,
            },
            series=[
                {
                    "name": "Roofline",
                    "type": "line",
                    "data": roofline_data,
                    "lineStyle": {"color": "#ee6666", "width": 2},
                    "symbol": "none",
                },
                {
                    "name": "Kernels",
                    "type": "scatter",
                    "data": scatter_data,
                    "symbolSize": 12,
                    "itemStyle": {"color": "#5470c6"},
                },
            ],
            extra={
                "graphic": [
                    {
                        "type": "text",
                        "left": "80%",
                        "top": "15%",
                        "style": {"text": f"Peak: {peak_compute:.0f} GFLOPS", "fontSize": 12},
                    },
                    {
                        "type": "text",
                        "left": "10%",
                        "top": "60%",
                        "style": {"text": f"BW: {peak_bandwidth:.0f} GB/s", "fontSize": 12},
                    },
                ]
            },
        )

    @classmethod
    def gauge_chart(cls,
                    value: float,
                    max_value: float = 100,
                    title: str = "",
                    unit: str = "%") -> EChartsOption:
        """
        创建仪表盘

        Args:
            value: 当前值
            max_value: 最大值
            title: 标题
            unit: 单位
        """
        return EChartsOption(
            title={"text": title} if title else {},
            series=[{
                "type": "gauge",
                "min": 0,
                "max": max_value,
                "progress": {"show": True, "width": 18},
                "axisLine": {"lineStyle": {"width": 18}},
                "axisTick": {"show": False},
                "splitLine": {"length": 15, "lineStyle": {"width": 2, "color": "#999"}},
                "axisLabel": {"distance": 25, "color": "#999", "fontSize": 12},
                "anchor": {"show": True, "showAbove": True, "size": 25, "itemStyle": {"borderWidth": 10}},
                "detail": {"valueAnimation": True, "fontSize": 24, "formatter": f"{{value}}{unit}"},
                "data": [{"value": value, "name": title}],
            }],
        )

    @classmethod
    def heatmap_chart(cls,
                      x_data: List[str],
                      y_data: List[str],
                      values: List[List[float]],
                      title: str = "") -> EChartsOption:
        """
        创建热力图

        Args:
            x_data: X 轴标签
            y_data: Y 轴标签
            values: [[x_idx, y_idx, value], ...]
            title: 标题
        """
        # 转换数据格式
        data = []
        for i, row in enumerate(values):
            for j, val in enumerate(row):
                data.append([j, i, val])

        max_val = max(max(row) for row in values) if values else 1

        return EChartsOption(
            title={"text": title} if title else {},
            tooltip={"position": "top"},
            xAxis={"type": "category", "data": x_data, "splitArea": {"show": True}},
            yAxis={"type": "category", "data": y_data, "splitArea": {"show": True}},
            extra={
                "visualMap": {
                    "min": 0,
                    "max": max_val,
                    "calculable": True,
                    "orient": "horizontal",
                    "left": "center",
                    "bottom": "5%",
                },
            },
            series=[{
                "name": title or "Heatmap",
                "type": "heatmap",
                "data": data,
                "label": {"show": True},
                "emphasis": {
                    "itemStyle": {"shadowBlur": 10, "shadowColor": "rgba(0, 0, 0, 0.5)"}
                },
            }],
        )

    @classmethod
    def sankey_chart(cls,
                     nodes: List[str],
                     links: List[Dict[str, Any]],
                     title: str = "") -> EChartsOption:
        """
        创建桑基图

        Args:
            nodes: 节点名称列表
            links: [{"source": str, "target": str, "value": float}, ...]
            title: 标题
        """
        return EChartsOption(
            title={"text": title} if title else {},
            tooltip={"trigger": "item", "triggerOn": "mousemove"},
            series=[{
                "type": "sankey",
                "emphasis": {"focus": "adjacency"},
                "data": [{"name": n} for n in nodes],
                "links": links,
                "lineStyle": {"color": "gradient", "curveness": 0.5},
            }],
        )

    @classmethod
    def from_tiling_result(cls, tiling_result, title: str = "Tiling Result") -> EChartsOption:
        """
        从 TilingResult 生成图表

        Args:
            tiling_result: TilingResult 对象
        """
        # 提取数据
        ops = []
        compute_cycles = []
        memory_cycles = []

        for op_result in tiling_result.op_results:
            ops.append(op_result.op_name)
            compute_cycles.append(op_result.compute_cycles)
            memory_cycles.append(op_result.memory_cycles)

        return cls.bar_chart(
            x_data=ops,
            series_data={
                "Compute": compute_cycles,
                "Memory": memory_cycles,
            },
            title=title,
            x_label="Operators",
            y_label="Cycles",
            stack=True,
        )

    @classmethod
    def from_compare_result(cls, compare_result, title: str = "Strategy Comparison") -> EChartsOption:
        """
        从 CompareResult 生成图表

        Args:
            compare_result: CompareResult 对象
        """
        strategies = []
        total_cycles = []
        utilization = []

        for strategy_name, result in compare_result.results.items():
            strategies.append(strategy_name)
            total_cycles.append(result.tiling_result.total_cycles)
            utilization.append(result.tiling_result.compute_utilization * 100)

        return cls.bar_chart(
            x_data=strategies,
            series_data={
                "Total Cycles": total_cycles,
            },
            title=title,
            x_label="Strategy",
            y_label="Cycles",
        )


# 便捷函数
def to_echarts(data: Any, chart_type: ChartType = ChartType.BAR, **kwargs) -> EChartsOption:
    """
    便捷转换函数

    Args:
        data: 输入数据
        chart_type: 图表类型
        **kwargs: 额外参数

    Returns:
        EChartsOption
    """
    converter = EChartsConverter

    if chart_type == ChartType.LINE:
        return converter.line_chart(**kwargs)
    elif chart_type == ChartType.BAR:
        return converter.bar_chart(**kwargs)
    elif chart_type == ChartType.PIE:
        return converter.pie_chart(**kwargs)
    elif chart_type == ChartType.SCATTER:
        return converter.scatter_chart(**kwargs)
    elif chart_type == ChartType.GAUGE:
        return converter.gauge_chart(**kwargs)
    elif chart_type == ChartType.HEATMAP:
        return converter.heatmap_chart(**kwargs)
    elif chart_type == ChartType.SANKEY:
        return converter.sankey_chart(**kwargs)
    else:
        raise ValueError(f"Unsupported chart type: {chart_type}")
