# Compare 可视化架构设计

> 设计时间: 2026-02-08
> 核心思想: 三层架构 + 误差传播分析

---

## 1. 架构概述

### 1.1 三层设计

```
┌─────────────────────────────────────────────────────────────┐
│                    模型级可视化                              │
│              (ModelVisualizer / EngineVisualizer)            │
│  ┌───────────────────────────────────────────────────────┐  │
│  │  误差传播图（算子链）                                  │  │
│  │  ├─ Conv1 → BN1 → ReLU1 → Pool1 → Conv2 → ...        │  │
│  │  │   15%    12%    10%     8%      25% ← 误差爆发点   │  │
│  │  │                                                     │  │
│  │  ├─ 多算子聚合分析                                     │  │
│  │  │   - 哪些算子误差大                                 │  │
│  │  │   - 误差累积路径                                   │  │
│  │  │   - 误差源头定位                                   │  │
│  │  │                                                     │  │
│  │  └─ 处理部分算子缺失（DUT 不保存所有输出）            │  │
│  └───────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                              ↑
                              │ 调用
                              │
┌─────────────────────────────────────────────────────────────┐
│                    策略级可视化                              │
│                  (Strategy.visualize())                      │
│  ┌──────────────────┬──────────────────┬─────────────────┐  │
│  │ BitAnalysis      │ Blocked          │ Fuzzy           │  │
│  │ ─────────────    │ ─────────────    │ ─────────────   │  │
│  │ 展示比对原理:    │ 展示比对原理:    │ 展示比对原理:   │  │
│  │ - Bit 语义分析   │ - 块级分析       │ - 统计指标      │  │
│  │ - Sign/Exp/Mant  │ - QSNR 分布      │ - Cosine/QSNR   │  │
│  │ - 格式理解       │ - 局部vs全局     │ - 容忍度阈值    │  │
│  └──────────────────┴──────────────────┴─────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                              ↑
                              │ 调用
                              │
┌─────────────────────────────────────────────────────────────┐
│                    基础底座                                  │
│                   (Visualizer)                               │
│  ┌───────────────────────────────────────────────────────┐  │
│  │  pyecharts 封装                                        │  │
│  │  ├─ 饼图、柱状图、热力图、雷达图、散点图、...          │  │
│  │  ├─ HTML 报告生成框架                                  │  │
│  │  ├─ 颜色方案、主题管理                                 │  │
│  │  └─ 通用工具函数                                       │  │
│  └───────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

---

## 2. 基础底座 (Visualizer)

### 2.1 职责

✅ 提供 pyecharts 的通用封装
✅ 管理颜色方案、主题
✅ HTML 报告生成框架
✅ 数据格式转换工具
❌ **不包含**业务逻辑（由策略层/模型层实现）

### 2.2 实现

```python
# aidevtools/compare/visualizer.py (新建)

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

    提供 pyecharts 封装，不包含业务逻辑
    """

    # 颜色方案
    COLORS = {
        'critical': '#FF4444',    # 红色 - CRITICAL
        'warning':  '#FFA500',    # 橙色 - WARNING
        'info':     '#FFFF00',    # 黄色 - INFO
        'ok':       '#44FF44',    # 绿色 - PASS
        'gray':     '#CCCCCC',    # 灰色 - 背景
    }

    @staticmethod
    def create_pie(
        data: Dict[str, float],
        config: ChartConfig = ChartConfig(),
        radius: List[str] = ["40%", "70%"],
    ) -> Pie:
        """创建饼图"""
        pie = (
            Pie(init_opts=opts.InitOpts(theme=config.theme))
            .add(
                "",
                list(data.items()),
                radius=radius,
            )
            .set_global_opts(
                title_opts=opts.TitleOpts(title=config.title),
                legend_opts=opts.LegendOpts(orient="vertical", pos_left="left"),
            )
            .set_series_opts(
                label_opts=opts.LabelOpts(formatter="{b}: {c} ({d}%)")
            )
        )
        return pie

    @staticmethod
    def create_bar(
        x_data: List[str],
        series_data: Dict[str, List[float]],
        config: ChartConfig = ChartConfig(),
        horizontal: bool = False,
        stack: bool = False,
    ) -> Bar:
        """创建柱状图"""
        bar = Bar(init_opts=opts.InitOpts(theme=config.theme))

        if horizontal:
            bar.add_yaxis("", x_data)
            for name, data in series_data.items():
                bar.add_xaxis(name, data, stack="stack" if stack else None)
        else:
            bar.add_xaxis(x_data)
            for name, data in series_data.items():
                bar.add_yaxis(name, data, stack="stack" if stack else None)

        bar.set_global_opts(
            title_opts=opts.TitleOpts(title=config.title),
            xaxis_opts=opts.AxisOpts(axislabel_opts=opts.LabelOpts(rotate=-15)),
        )
        return bar

    @staticmethod
    def create_heatmap(
        x_data: List[str],
        y_data: List[str],
        values: List[List[float]],
        config: ChartConfig = ChartConfig(),
        max_val: Optional[float] = None,
    ) -> HeatMap:
        """创建热力图"""
        # 转换数据格式 [[x_idx, y_idx, value], ...]
        data = []
        for i, row in enumerate(values):
            for j, val in enumerate(row):
                data.append([j, i, val])

        if max_val is None:
            max_val = max(max(row) for row in values) if values else 1

        heatmap = (
            HeatMap(init_opts=opts.InitOpts(theme=config.theme))
            .add_xaxis(x_data)
            .add_yaxis("", y_data, data)
            .set_global_opts(
                title_opts=opts.TitleOpts(title=config.title),
                visualmap_opts=opts.VisualMapOpts(
                    min_=0,
                    max_=max_val,
                    calculable=True,
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
        config: ChartConfig = ChartConfig(),
    ) -> Radar:
        """创建雷达图"""
        radar = Radar(init_opts=opts.InitOpts(theme=config.theme))
        radar.add_schema([opts.RadarIndicatorItem(**s) for s in schema])

        for name, data in series_data.items():
            radar.add(name, [data])

        radar.set_global_opts(title_opts=opts.TitleOpts(title=config.title))
        return radar

    @staticmethod
    def create_sankey(
        nodes: List[str],
        links: List[Dict[str, Any]],
        config: ChartConfig = ChartConfig(),
    ) -> Sankey:
        """创建桑基图（误差传播）"""
        sankey = (
            Sankey(init_opts=opts.InitOpts(theme=config.theme))
            .add(
                "",
                nodes=[{"name": n} for n in nodes],
                links=links,
                linestyle_opt=opts.LineStyleOpts(opacity=0.5, curve=0.5, color="source"),
            )
            .set_global_opts(title_opts=opts.TitleOpts(title=config.title))
        )
        return sankey

    @staticmethod
    def create_line(
        x_data: List[Any],
        series_data: Dict[str, List[float]],
        config: ChartConfig = ChartConfig(),
        smooth: bool = True,
    ) -> Line:
        """创建折线图"""
        line = Line(init_opts=opts.InitOpts(theme=config.theme))
        line.add_xaxis(x_data)

        for name, data in series_data.items():
            line.add_yaxis(name, data, is_smooth=smooth)

        line.set_global_opts(
            title_opts=opts.TitleOpts(title=config.title),
            tooltip_opts=opts.TooltipOpts(trigger="axis"),
        )
        return line

    @staticmethod
    def create_page(title: str = "Report") -> Page:
        """创建 Page（多图表容器）"""
        return Page(page_title=title, layout=Page.SimplePageLayout)

    @staticmethod
    def render_html(page: Page, output_path: str) -> None:
        """渲染 HTML 文件"""
        page.render(output_path)
```

---

## 3. 策略级可视化

### 3.1 设计原则

**关键**: 可视化要**体现策略的比对原理**，而不是简单画图

### 3.2 BitAnalysisStrategy 可视化

**比对原理**:
- 理解浮点格式 (sign | exponent | mantissa)
- 区分错误类型的语义
- Sign flip = 符号错误（最严重）
- Exponent diff = 数量级错误（严重）
- Mantissa diff = 精度误差（可接受）

**可视化体现**:

```python
# aidevtools/compare/strategy/bit_analysis.py

class BitAnalysisStrategy:
    # ... 现有方法 ...

    @staticmethod
    def visualize(
        result: BitAnalysisResult,
        golden: Optional[np.ndarray] = None,
        dut: Optional[np.ndarray] = None,
    ) -> Page:
        """
        BitAnalysis 策略级可视化

        体现比对原理:
        - 展示 bit 级语义分析
        - 区分 sign/exponent/mantissa 错误
        - 突出格式理解
        """
        from aidevtools.compare.visualizer import Visualizer, ChartConfig

        page = Visualizer.create_page(title="Bit Analysis Report")

        # 1. 错误类型分布饼图 (体现语义分析)
        error_data = {
            "✅ No Diff": result.summary.total_elements - result.summary.diff_elements,
            "🟡 Mantissa Only (精度误差)": result.summary.mantissa_diff_count,
            "🟠 Exponent Diff (数量级错误)": result.summary.exponent_diff_count,
            "🔴 Sign Flip (符号错误)": result.summary.sign_flip_count,
        }
        pie = Visualizer.create_pie(
            error_data,
            ChartConfig(title=f"Error Type Distribution ({result.fmt.name})"),
        )
        page.add(pie)

        # 2. Bit 布局图 (体现格式理解)
        # 显示 FP32: S|EEEEEEEE|MMMMMMMMMMMMMMMMMMMMMMM
        #             1   8           23
        bit_layout_chart = BitAnalysisStrategy._create_bit_layout_chart(result)
        page.add(bit_layout_chart)

        # 3. 指数偏移分布 (体现数量级分析)
        if result.summary.exponent_diff_count > 0:
            exp_dist_chart = BitAnalysisStrategy._create_exponent_dist_chart(result, golden, dut)
            page.add(exp_dist_chart)

        # 4. 告警摘要 (体现严重度分级)
        warnings_chart = BitAnalysisStrategy._create_warnings_chart(result)
        page.add(warnings_chart)

        return page

    @staticmethod
    def _create_bit_layout_chart(result: BitAnalysisResult) -> Bar:
        """
        Bit 布局图

        展示格式理解: FP32 = S | E8 | M23
        """
        from aidevtools.compare.visualizer import Visualizer, ChartConfig

        # X 轴: bit regions
        x_data = ["Sign (b31)", f"Exponent (b30-b{31-result.fmt.exponent_bits+1})",
                  f"Mantissa (b{31-result.fmt.exponent_bits}-b0)"]

        # Y 轴: 错误计数
        error_counts = {
            "Error Count": [
                result.summary.sign_flip_count,
                result.summary.exponent_diff_count,
                result.summary.mantissa_diff_count,
            ]
        }

        bar = Visualizer.create_bar(
            x_data,
            error_counts,
            ChartConfig(title=f"Bit Layout Analysis ({result.fmt.name})"),
        )
        return bar

    @staticmethod
    def _create_exponent_dist_chart(
        result: BitAnalysisResult,
        golden: np.ndarray,
        dut: np.ndarray,
    ) -> Bar:
        """
        指数偏移分布

        体现数量级分析:
        - 偏移 1 bit → 2x 误差
        - 偏移 2 bit → 4x 误差
        - 偏移 10 bit → 1024x 误差
        """
        # TODO: 需要重新计算指数偏移分布
        # 这里需要访问原始数据
        pass

    @staticmethod
    def _create_warnings_chart(result: BitAnalysisResult) -> Bar:
        """
        告警摘要柱状图

        体现严重度分级: CRITICAL > WARNING > INFO
        """
        from aidevtools.compare.visualizer import Visualizer, ChartConfig

        # 统计各级别告警数量
        critical_count = sum(1 for w in result.warnings if w.level == WarnLevel.CRITICAL)
        warning_count = sum(1 for w in result.warnings if w.level == WarnLevel.WARNING)
        info_count = sum(1 for w in result.warnings if w.level == WarnLevel.INFO)

        x_data = ["🔴 CRITICAL", "🟠 WARNING", "🟡 INFO"]
        counts = {"Count": [critical_count, warning_count, info_count]}

        bar = Visualizer.create_bar(
            x_data,
            counts,
            ChartConfig(title="Warning Summary"),
        )
        return bar
```

---

### 3.3 BlockedStrategy 可视化

**比对原理**:
- 块级分析（局部 vs 全局）
- QSNR 衡量信噪比
- 识别局部异常块

**可视化体现**:

```python
# aidevtools/compare/strategy/blocked.py

class BlockedStrategy:
    # ... 现有方法 ...

    @staticmethod
    def visualize(blocks: List[BlockResult], threshold: float = 20.0) -> Page:
        """
        Blocked 策略级可视化

        体现比对原理:
        - 展示块级分析（局部vs全局）
        - QSNR 分布特征
        - 识别异常块
        """
        from aidevtools.compare.visualizer import Visualizer, ChartConfig

        page = Visualizer.create_page(title="Blocked Analysis Report")

        # 1. QSNR 热力图 (体现空间分布)
        # 假设 8 列布局
        cols = 8
        rows = (len(blocks) + cols - 1) // cols
        qsnr_matrix = BlockedStrategy._build_qsnr_matrix(blocks, rows, cols)

        heatmap = Visualizer.create_heatmap(
            x_data=[f"Col {i}" for i in range(cols)],
            y_data=[f"Row {i}" for i in range(rows)],
            values=qsnr_matrix,
            config=ChartConfig(title="Block QSNR Heatmap"),
        )
        page.add(heatmap)

        # 2. QSNR 分布直方图 (体现统计特征)
        qsnr_dist_chart = BlockedStrategy._create_qsnr_distribution(blocks, threshold)
        page.add(qsnr_dist_chart)

        # 3. 异常块详情 (体现局部分析)
        failed_blocks = [b for b in blocks if not b.passed]
        if failed_blocks:
            failed_chart = BlockedStrategy._create_failed_blocks_chart(failed_blocks)
            page.add(failed_chart)

        return page

    @staticmethod
    def _build_qsnr_matrix(blocks: List[BlockResult], rows: int, cols: int) -> List[List[float]]:
        """构建 QSNR 矩阵"""
        matrix = [[0.0] * cols for _ in range(rows)]
        for idx, block in enumerate(blocks):
            r = idx // cols
            c = idx % cols
            if r < rows and c < cols:
                matrix[r][c] = block.qsnr
        return matrix

    @staticmethod
    def _create_qsnr_distribution(blocks: List[BlockResult], threshold: float) -> Bar:
        """QSNR 分布直方图"""
        from aidevtools.compare.visualizer import Visualizer, ChartConfig

        # 分箱统计
        bins = [0, 10, 20, 30, 40, 50, 100]
        counts = [0] * (len(bins) - 1)
        for block in blocks:
            for i in range(len(bins) - 1):
                if bins[i] <= block.qsnr < bins[i+1]:
                    counts[i] += 1
                    break

        x_data = [f"[{bins[i]}-{bins[i+1]})" for i in range(len(bins)-1)]
        series = {"Block Count": counts}

        bar = Visualizer.create_bar(
            x_data,
            series,
            ChartConfig(title=f"QSNR Distribution (threshold={threshold} dB)"),
        )
        return bar

    @staticmethod
    def _create_failed_blocks_chart(failed_blocks: List[BlockResult]) -> Bar:
        """失败块详情"""
        from aidevtools.compare.visualizer import Visualizer, ChartConfig

        # 前 10 个失败块
        top_failed = sorted(failed_blocks, key=lambda b: b.qsnr)[:10]

        x_data = [f"Block {b.offset//b.size}" for b in top_failed]
        series = {"QSNR (dB)": [b.qsnr for b in top_failed]}

        bar = Visualizer.create_bar(
            x_data,
            series,
            ChartConfig(title="Top 10 Failed Blocks"),
        )
        return bar
```

---

### 3.4 FuzzyStrategy 可视化

**比对原理**:
- 统计指标（Cosine, QSNR）
- 容忍度阈值

**可视化体现**:

```python
# aidevtools/compare/strategy/fuzzy.py

class FuzzyStrategy:
    # ... 现有方法 ...

    @staticmethod
    def visualize(result: FuzzyResult, config: FuzzyConfig) -> Page:
        """
        Fuzzy 策略级可视化

        体现比对原理:
        - 展示统计指标
        - 突出阈值判定
        """
        from aidevtools.compare.visualizer import Visualizer, ChartConfig

        page = Visualizer.create_page(title="Fuzzy Analysis Report")

        # 1. 指标雷达图 (体现多指标综合)
        schema = [
            {"name": "Cosine", "max": 1.0},
            {"name": "QSNR", "max": 50.0},
            {"name": "Max Abs", "max": config.threshold_abs},
            {"name": "Max Relative", "max": config.threshold_rel},
        ]

        series_data = {
            "Metrics": [
                result.cosine_similarity,
                result.qsnr,
                result.max_abs_diff,
                result.max_relative_diff,
            ]
        }

        radar = Visualizer.create_radar(schema, series_data, ChartConfig(title="Fuzzy Metrics"))
        page.add(radar)

        # 2. 阈值对比柱状图 (体现判定规则)
        threshold_chart = FuzzyStrategy._create_threshold_chart(result, config)
        page.add(threshold_chart)

        return page

    @staticmethod
    def _create_threshold_chart(result: FuzzyResult, config: FuzzyConfig) -> Bar:
        """阈值对比柱状图"""
        from aidevtools.compare.visualizer import Visualizer, ChartConfig

        x_data = ["Cosine", "QSNR", "Max Abs", "Max Rel"]

        actual = [
            result.cosine_similarity,
            result.qsnr,
            result.max_abs_diff,
            result.max_relative_diff,
        ]

        threshold = [
            config.threshold_cosine,
            config.threshold_qsnr,
            config.threshold_abs,
            config.threshold_rel,
        ]

        series = {
            "Actual": actual,
            "Threshold": threshold,
        }

        bar = Visualizer.create_bar(x_data, series, ChartConfig(title="Metrics vs Threshold"))
        return bar
```

---

## 4. 模型级可视化 (核心)

### 4.1 设计原则

**关键需求**:
1. **误差传播分析** - 跨算子的误差累积
2. **误差源头定位** - 哪些算子是误差源
3. **处理缺失数据** - DUT 不保存所有算子输出

### 4.2 ModelVisualizer 实现

```python
# aidevtools/compare/model_visualizer.py (新建)

from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from enum import Enum
import numpy as np


class OpStatus(Enum):
    """算子状态"""
    HAS_DATA = "has_data"      # 有完整数据
    MISSING_DUT = "missing_dut"  # 缺少 DUT 输出
    SKIPPED = "skipped"        # 未比对


@dataclass
class OpCompareResult:
    """单算子比对结果"""
    op_name: str
    op_id: int
    status: OpStatus

    # 比对结果（如果有数据）
    qsnr: Optional[float] = None
    cosine: Optional[float] = None
    max_abs: Optional[float] = None
    passed: Optional[bool] = None

    # 策略结果（可选）
    exact_result: Optional[Any] = None
    fuzzy_result: Optional[Any] = None
    bitwise_result: Optional[Any] = None


@dataclass
class ModelCompareResult:
    """模型级比对结果"""
    model_name: str
    ops: List[OpCompareResult]

    # 全局统计
    total_ops: int
    ops_with_data: int
    ops_missing_dut: int
    passed_ops: int
    failed_ops: int


class ModelVisualizer:
    """
    模型级可视化

    核心功能:
    1. 误差传播分析
    2. 跨算子聚合
    3. 处理缺失数据
    """

    @staticmethod
    def visualize(result: ModelCompareResult) -> Page:
        """
        生成模型级完整报告

        包含:
        - 误差传播 Sankey 图
        - 算子 QSNR 排序
        - 误差累积曲线
        - 缺失数据标注
        """
        from aidevtools.compare.visualizer import Visualizer, ChartConfig

        page = Visualizer.create_page(title=f"Model Analysis: {result.model_name}")

        # 1. 误差传播 Sankey 图
        sankey = ModelVisualizer._create_error_propagation_sankey(result)
        page.add(sankey)

        # 2. 算子 QSNR 排序（找出瓶颈）
        qsnr_chart = ModelVisualizer._create_op_qsnr_ranking(result)
        page.add(qsnr_chart)

        # 3. 误差累积曲线
        accumulation_chart = ModelVisualizer._create_error_accumulation(result)
        page.add(accumulation_chart)

        # 4. 数据完整性摘要
        completeness_chart = ModelVisualizer._create_data_completeness(result)
        page.add(completeness_chart)

        return page

    @staticmethod
    def _create_error_propagation_sankey(result: ModelCompareResult) -> Sankey:
        """
        误差传播 Sankey 图

        展示误差在算子间的流动:
        Input → Conv1 → BN1 → ReLU1 → Conv2 → ...
               (15%)   (12%)   (10%)    (25%) ← 误差爆发点
        """
        from aidevtools.compare.visualizer import Visualizer, ChartConfig

        # 构建节点
        nodes = ["Input"]
        for op in result.ops:
            if op.status == OpStatus.HAS_DATA:
                nodes.append(op.op_name)
        nodes.append("Output")

        # 构建链接（误差流动）
        links = []
        prev_qsnr = 100.0  # Input 假设无误差

        for i, op in enumerate(result.ops):
            if op.status == OpStatus.HAS_DATA and op.qsnr is not None:
                # 误差 = 100 - QSNR (简化)
                error_rate = 100 - op.qsnr

                links.append({
                    "source": nodes[i],
                    "target": op.op_name,
                    "value": error_rate,
                })

                prev_qsnr = op.qsnr
            elif op.status == OpStatus.MISSING_DUT:
                # 缺失数据用虚线表示
                links.append({
                    "source": nodes[i] if i > 0 else "Input",
                    "target": op.op_name + " (missing)",
                    "value": 0,
                })

        sankey = Visualizer.create_sankey(
            nodes=nodes,
            links=links,
            config=ChartConfig(title="Error Propagation Flow"),
        )
        return sankey

    @staticmethod
    def _create_op_qsnr_ranking(result: ModelCompareResult) -> Bar:
        """
        算子 QSNR 排序

        找出误差最大的算子（瓶颈定位）
        """
        from aidevtools.compare.visualizer import Visualizer, ChartConfig

        # 提取有 QSNR 数据的算子
        ops_with_qsnr = [
            (op.op_name, op.qsnr)
            for op in result.ops
            if op.status == OpStatus.HAS_DATA and op.qsnr is not None
        ]

        # 按 QSNR 升序排序（误差大的在前）
        ops_with_qsnr.sort(key=lambda x: x[1])

        # 取前 20 个
        top_ops = ops_with_qsnr[:20]

        x_data = [name for name, _ in top_ops]
        series = {"QSNR (dB)": [qsnr for _, qsnr in top_ops]}

        bar = Visualizer.create_bar(
            x_data,
            series,
            ChartConfig(title="Op QSNR Ranking (Lower = Worse)"),
            horizontal=True,
        )
        return bar

    @staticmethod
    def _create_error_accumulation(result: ModelCompareResult) -> Line:
        """
        误差累积曲线

        展示误差随算子层数的累积趋势
        """
        from aidevtools.compare.visualizer import Visualizer, ChartConfig

        # X 轴: 算子序号
        x_data = []
        y_qsnr = []
        y_cosine = []

        for i, op in enumerate(result.ops):
            if op.status == OpStatus.HAS_DATA:
                x_data.append(f"{i}: {op.op_name}")
                y_qsnr.append(op.qsnr if op.qsnr else 0)
                y_cosine.append(op.cosine if op.cosine else 0)

        series = {
            "QSNR": y_qsnr,
            "Cosine": y_cosine,
        }

        line = Visualizer.create_line(
            x_data,
            series,
            ChartConfig(title="Error Accumulation Across Ops"),
        )
        return line

    @staticmethod
    def _create_data_completeness(result: ModelCompareResult) -> Pie:
        """
        数据完整性摘要

        展示有多少算子缺失 DUT 输出
        """
        from aidevtools.compare.visualizer import Visualizer, ChartConfig

        data = {
            "✅ Has Data": result.ops_with_data,
            "❌ Missing DUT": result.ops_missing_dut,
            "⏭️ Skipped": result.total_ops - result.ops_with_data - result.ops_missing_dut,
        }

        pie = Visualizer.create_pie(
            data,
            ChartConfig(title=f"Data Completeness ({result.ops_with_data}/{result.total_ops})"),
        )
        return pie
```

---

### 4.3 EngineVisualizer (综合报告)

```python
# aidevtools/compare/engine_visualizer.py (新建)

class EngineVisualizer:
    """
    Engine 级可视化

    综合多策略 + 多算子的完整报告
    """

    @staticmethod
    def create_comprehensive_report(
        engine_result: Dict[str, Any],
        model_result: Optional[ModelCompareResult] = None,
    ) -> Page:
        """
        生成综合报告

        包含:
        1. 总览 (状态判定)
        2. 各策略摘要
        3. 模型级分析 (如果有)
        4. 决策建议
        """
        from aidevtools.compare.visualizer import Visualizer, ChartConfig

        page = Visualizer.create_page(title="Comprehensive Compare Report")

        # 1. 总览状态
        status_chart = EngineVisualizer._create_status_overview(engine_result)
        page.add(status_chart)

        # 2. 策略摘要雷达图
        strategy_radar = EngineVisualizer._create_strategy_summary(engine_result)
        page.add(strategy_radar)

        # 3. 模型级分析 (如果有数据)
        if model_result:
            model_page = ModelVisualizer.visualize(model_result)
            # 合并 page
            for chart in model_page.charts:
                page.add(chart)

        # 4. 决策建议表格
        decision_chart = EngineVisualizer._create_decision_guide(engine_result)
        page.add(decision_chart)

        return page

    @staticmethod
    def _create_status_overview(engine_result: Dict[str, Any]) -> Pie:
        """总览状态饼图"""
        status = engine_result.get('status', 'UNKNOWN')

        # 状态分布（简化示例）
        data = {
            "✅ PASS": 1 if status == "PASS" else 0,
            "⚠️ GOLDEN_SUSPECT": 1 if status == "GOLDEN_SUSPECT" else 0,
            "🔴 DUT_ISSUE": 1 if status == "DUT_ISSUE" else 0,
            "🟠 BOTH_SUSPECT": 1 if status == "BOTH_SUSPECT" else 0,
        }

        from aidevtools.compare.visualizer import Visualizer, ChartConfig
        pie = Visualizer.create_pie(data, ChartConfig(title=f"Overall Status: {status}"))
        return pie

    @staticmethod
    def _create_strategy_summary(engine_result: Dict[str, Any]) -> Radar:
        """策略摘要雷达图"""
        # 提取各策略结果
        exact = engine_result.get('exact')
        fuzzy = engine_result.get('fuzzy_pure')
        bitwise = engine_result.get('bitwise')

        schema = [
            {"name": "Exact Pass", "max": 1},
            {"name": "Fuzzy QSNR", "max": 50},
            {"name": "Fuzzy Cosine", "max": 1},
            {"name": "Bitwise OK", "max": 1},
        ]

        series_data = {
            "Metrics": [
                1 if exact and exact.passed else 0,
                fuzzy.qsnr if fuzzy else 0,
                fuzzy.cosine_similarity if fuzzy else 0,
                1 if bitwise and not bitwise.has_critical else 0,
            ]
        }

        from aidevtools.compare.visualizer import Visualizer, ChartConfig
        radar = Visualizer.create_radar(schema, series_data, ChartConfig(title="Strategy Summary"))
        return radar

    @staticmethod
    def _create_decision_guide(engine_result: Dict[str, Any]) -> Bar:
        """决策建议"""
        # TODO: 根据 status 给出建议
        pass
```

---

## 5. 处理 DUT 缺失数据

### 5.1 设计方案

**问题**: DUT 不保存所有算子输出

**解决方案**:

1. **标注缺失**:
   ```python
   @dataclass
   class OpCompareResult:
       status: OpStatus  # HAS_DATA / MISSING_DUT / SKIPPED
   ```

2. **可视化区分**:
   ```python
   # Sankey 图中用虚线表示缺失
   if op.status == OpStatus.MISSING_DUT:
       links.append({
           "source": prev_op,
           "target": op.op_name + " ⚠️ (missing)",
           "value": 0,
           "lineStyle": {"type": "dashed"},  # 虚线
       })
   ```

3. **插值估算** (可选):
   ```python
   # 如果 Op1 和 Op3 有数据，Op2 缺失
   # 可以用线性插值估算 Op2 的误差
   if op2.status == OpStatus.MISSING_DUT:
       op2.estimated_qsnr = (op1.qsnr + op3.qsnr) / 2
   ```

4. **统计报告**:
   ```python
   # 数据完整性饼图
   data = {
       "✅ Has Data": 45,
       "❌ Missing DUT": 12,
       "⏭️ Skipped": 3,
   }
   ```

---

## 6. 使用示例

### 6.1 策略级可视化

```python
# BitAnalysis
result = BitAnalysisStrategy.compare(golden, dut, fmt=FP32)
page = BitAnalysisStrategy.visualize(result, golden, dut)
page.render("bitwise_report.html")

# Blocked
blocks = BlockedStrategy.compare(golden, dut, block_size=1024)
page = BlockedStrategy.visualize(blocks, threshold=20.0)
page.render("blocked_report.html")
```

### 6.2 模型级可视化

```python
# 构建模型比对结果
ops = [
    OpCompareResult("conv1", 0, OpStatus.HAS_DATA, qsnr=45.2, passed=True),
    OpCompareResult("bn1", 1, OpStatus.HAS_DATA, qsnr=42.1, passed=True),
    OpCompareResult("relu1", 2, OpStatus.MISSING_DUT),  # 缺失
    OpCompareResult("pool1", 3, OpStatus.HAS_DATA, qsnr=38.5, passed=True),
    OpCompareResult("conv2", 4, OpStatus.HAS_DATA, qsnr=12.3, passed=False),  # 瓶颈
]

model_result = ModelCompareResult(
    model_name="ResNet50",
    ops=ops,
    total_ops=5,
    ops_with_data=4,
    ops_missing_dut=1,
    passed_ops=3,
    failed_ops=1,
)

# 生成模型级报告
page = ModelVisualizer.visualize(model_result)
page.render("model_report.html")
```

### 6.3 Engine 综合报告

```python
# 运行 Engine
engine = CompareEngine.standard()
engine_result = engine.run(dut=dut, golden=golden, golden_qnt=golden_qnt)

# 如果有模型级数据
model_result = build_model_result_from_records()

# 生成综合报告
page = EngineVisualizer.create_comprehensive_report(engine_result, model_result)
page.render("comprehensive_report.html")
```

---

## 7. 总结

### 7.1 架构优势

✅ **三层清晰**:
- 基础底座：复用性强
- 策略级：体现比对原理
- 模型级：误差传播分析

✅ **体现原理**:
- 不是简单画图
- 可视化反映策略设计思想

✅ **处理现实约束**:
- DUT 部分数据缺失
- 标注 + 估算 + 统计

### 7.2 工作量估算

| 层级 | 工作量 | 优先级 |
|------|--------|--------|
| 基础底座 (Visualizer) | 1.5h | P0 |
| BitAnalysis 可视化 | 1h | P0 |
| Blocked 可视化 | 1h | P0 |
| Fuzzy 可视化 | 0.5h | P1 |
| ModelVisualizer | 2h | P0 |
| EngineVisualizer | 1h | P1 |
| **总计 (P0)** | **5.5h** | - |
| **总计 (P0+P1)** | **7h** | - |

### 7.3 下一步

1. 实现基础底座 (1.5h)
2. 实现 BitAnalysis + Blocked 可视化 (2h)
3. 实现 ModelVisualizer (2h)
4. 测试和完善 (1h)

---

生成时间: 2026-02-08
