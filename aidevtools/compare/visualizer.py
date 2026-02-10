"""
Compare 可视化基础底座

提供 pyecharts 封装，无业务逻辑
"""

from pyecharts.charts import Pie, Bar, HeatMap, Radar, Scatter, Line, Sankey, Page
from pyecharts import options as opts
from pyecharts.globals import ThemeType
from typing import List, Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class ChartConfig:
    """图表配置"""
    title: str = ""
    width: str = "100%"
    height: str = "400px"
    theme: ThemeType = ThemeType.LIGHT


class Visualizer:
    """
    可视化基础底座

    纯工具类，提供 pyecharts 封装
    """

    # 标准颜色方案
    COLORS = {
        'critical': '#FF4444',    # 红色
        'warning':  '#FFA500',    # 橙色
        'info':     '#FFFF00',    # 黄色
        'ok':       '#44FF44',    # 绿色
        'gray':     '#CCCCCC',
    }

    @staticmethod
    def create_page(title: str = "Report") -> Page:
        """创建 Page"""
        return Page(page_title=title, layout=Page.SimplePageLayout)

    @staticmethod
    def create_pie(
        data: Dict[str, float],
        title: str = "",
        radius: List[str] = None,
    ) -> Pie:
        """创建饼图"""
        if radius is None:
            radius = ["40%", "70%"]

        pie = (
            Pie(init_opts=opts.InitOpts(theme=ThemeType.LIGHT))
            .add("", list(data.items()), radius=radius)
            .set_global_opts(
                title_opts=opts.TitleOpts(title=title),
                legend_opts=opts.LegendOpts(orient="vertical", pos_left="left"),
            )
            .set_series_opts(label_opts=opts.LabelOpts(formatter="{b}: {c} ({d}%)"))
        )
        return pie

    @staticmethod
    def create_bar(
        x_data: List[str],
        series_data: Dict[str, List[float]],
        title: str = "",
        horizontal: bool = False,
        stack: bool = False,
    ) -> Bar:
        """创建柱状图"""
        bar = Bar(init_opts=opts.InitOpts(theme=ThemeType.LIGHT))

        if horizontal:
            bar.add_xaxis(x_data)
            for name, data in series_data.items():
                bar.add_yaxis(name, data, stack="stack" if stack else None)
            bar.reversal_axis()
        else:
            bar.add_xaxis(x_data)
            for name, data in series_data.items():
                bar.add_yaxis(name, data, stack="stack" if stack else None)

        bar.set_global_opts(
            title_opts=opts.TitleOpts(title=title),
            xaxis_opts=opts.AxisOpts(axislabel_opts=opts.LabelOpts(rotate=-15)),
            toolbox_opts=opts.ToolboxOpts(feature=opts.ToolBoxFeatureOpts(
                save_as_image=opts.ToolBoxFeatureSaveAsImageOpts(title="Save")
            )),
        )
        return bar

    @staticmethod
    def create_heatmap(
        x_data: List[str],
        y_data: List[str],
        values: List[List[float]],
        title: str = "",
        max_val: Optional[float] = None,
    ) -> HeatMap:
        """创建热力图"""
        # 转换数据格式
        data = []
        for i, row in enumerate(values):
            for j, val in enumerate(row):
                data.append([j, i, val or 0])

        if max_val is None and values:
            max_val = max(max(row) for row in values if row)
            if max_val == 0:
                max_val = 1

        heatmap = (
            HeatMap(init_opts=opts.InitOpts(theme=ThemeType.LIGHT))
            .add_xaxis(x_data)
            .add_yaxis("", y_data, data)
            .set_global_opts(
                title_opts=opts.TitleOpts(title=title),
                visualmap_opts=opts.VisualMapOpts(
                    min_=0,
                    max_=max_val or 1,
                    is_calculable=True,
                    orient="horizontal",
                    pos_left="center",
                    pos_bottom="5%",
                ),
            )
        )
        return heatmap

    @staticmethod
    def create_radar(
        schema: List[Dict[str, Any]],
        series_data: Dict[str, List[float]],
        title: str = "",
    ) -> Radar:
        """创建雷达图"""
        radar = Radar(init_opts=opts.InitOpts(theme=ThemeType.LIGHT))
        radar.add_schema([opts.RadarIndicatorItem(**s) for s in schema])

        for name, data in series_data.items():
            radar.add(name, [data])

        radar.set_global_opts(title_opts=opts.TitleOpts(title=title))
        return radar

    @staticmethod
    def create_sankey(
        nodes: List[str],
        links: List[Dict[str, Any]],
        title: str = "",
    ) -> Sankey:
        """创建桑基图（误差传播）"""
        sankey = (
            Sankey(init_opts=opts.InitOpts(theme=ThemeType.LIGHT))
            .add(
                "",
                nodes=[{"name": n} for n in nodes],
                links=links,
                linestyle_opt=opts.LineStyleOpts(opacity=0.5, curve=0.5, color="source"),
            )
            .set_global_opts(
                title_opts=opts.TitleOpts(title=title),
                tooltip_opts=opts.TooltipOpts(trigger="item", trigger_on="mousemove"),
            )
        )
        return sankey

    @staticmethod
    def create_line(
        x_data: List[Any],
        series_data: Dict[str, List[float]],
        title: str = "",
        smooth: bool = True,
    ) -> Line:
        """创建折线图"""
        line = Line(init_opts=opts.InitOpts(theme=ThemeType.LIGHT))
        line.add_xaxis(x_data)

        for name, data in series_data.items():
            line.add_yaxis(name, data, is_smooth=smooth)

        line.set_global_opts(
            title_opts=opts.TitleOpts(title=title),
            tooltip_opts=opts.TooltipOpts(trigger="axis"),
            xaxis_opts=opts.AxisOpts(axislabel_opts=opts.LabelOpts(rotate=-15)),
        )
        return line

    @staticmethod
    def render_html(page: Page, output_path: str) -> None:
        """渲染 HTML 文件"""
        page.render(output_path)
