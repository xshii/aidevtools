# Compare 可视化快速开始

> 三层架构：基础底座 → 策略级 → 模型级

---

## 安装依赖

```bash
pip install pyecharts
```

---

## 1. 基础底座 (Visualizer)

纯工具类，提供 pyecharts 封装。

```python
from aidevtools.compare.visualizer import Visualizer

# 创建饼图
pie = Visualizer.create_pie(
    data={"A": 100, "B": 50, "C": 25},
    title="Distribution"
)

# 创建柱状图
bar = Visualizer.create_bar(
    x_data=["Cat1", "Cat2", "Cat3"],
    series_data={"Series1": [10, 20, 30]},
    title="Bar Chart"
)

# 创建 Page（多图表容器）
page = Visualizer.create_page(title="My Report")
page.add(pie)
page.add(bar)

# 渲染 HTML
Visualizer.render_html(page, "report.html")
```

---

## 2. 策略级可视化

每个策略根据自己的原理生成可视化。

### 2.1 BitAnalysisStrategy

**体现原理**: bit 语义分析 (sign/exponent/mantissa)

```python
from aidevtools.compare.strategy import BitAnalysisStrategy, FP32
import numpy as np

# 比对
golden = np.random.randn(1000).astype(np.float32)
result = golden + np.random.randn(1000) * 0.01

res = BitAnalysisStrategy.compare(golden, result, fmt=FP32)

# 可视化
page = BitAnalysisStrategy.visualize(res)
page.render("bitwise_report.html")
```

**生成图表**:
1. 错误类型饼图 (Sign Flip / Exponent / Mantissa)
2. Bit 布局柱状图 (S|E8|M23)
3. 告警摘要 (CRITICAL / WARNING / INFO)

---

### 2.2 BlockedStrategy

**体现原理**: 块级分析（局部 vs 全局）

```python
from aidevtools.compare.strategy import BlockedStrategy
import numpy as np

# 比对
golden = np.random.randn(10000).astype(np.float32)
result = golden + np.random.randn(10000) * 0.01

blocks = BlockedStrategy.compare(golden, result, block_size=1024)

# 可视化
page = BlockedStrategy.visualize(blocks, threshold=20.0, cols=8)
page.render("blocked_report.html")
```

**生成图表**:
1. QSNR 热力图 (空间分布)
2. QSNR 分布直方图 (统计特征)
3. 失败块详情 (Top 10)

---

## 3. 模型级可视化

误差传播分析，跨算子聚合。

```python
from aidevtools.compare.model_visualizer import (
    ModelVisualizer,
    ModelCompareResult,
    OpCompareResult,
    OpStatus,
)

# 构建模型结果
ops = [
    OpCompareResult("conv1", 0, OpStatus.HAS_DATA, qsnr=45.2, passed=True),
    OpCompareResult("bn1", 1, OpStatus.HAS_DATA, qsnr=42.1, passed=True),
    OpCompareResult("relu1", 2, OpStatus.MISSING_DUT),  # DUT 未保存
    OpCompareResult("conv2", 3, OpStatus.HAS_DATA, qsnr=12.3, passed=False),  # 瓶颈
]

model_result = ModelCompareResult(
    model_name="ResNet50",
    ops=ops,
    total_ops=4,
    ops_with_data=3,
    ops_missing_dut=1,
    passed_ops=2,
    failed_ops=1,
)

# 可视化
page = ModelVisualizer.visualize(model_result)
page.render("model_report.html")
```

**生成图表**:
1. 误差传播 Sankey 图 (Input → Op1 → Op2 → ...)
2. 算子 QSNR 排序 (找瓶颈)
3. 误差累积曲线 (趋势)
4. 数据完整性饼图 (Has Data / Missing DUT)

---

## 4. 运行 Demo

```bash
# 安装依赖
pip install pyecharts

# 运行示例
python demos/compare/demo_09_visualization.py

# 输出:
# ✅ HTML report saved: /tmp/bitwise_report.html
# ✅ HTML report saved: /tmp/blocked_report.html
# ✅ HTML report saved: /tmp/model_report.html
```

打开生成的 HTML 文件即可查看交互式报告。

---

## 5. 架构扩展

### 添加新策略的可视化

1. 在策略类中添加 `visualize()` 静态方法
2. 使用 `Visualizer` 创建图表
3. 返回 `Page` 对象

```python
class MyStrategy(CompareStrategy):
    @staticmethod
    def visualize(result) -> Page:
        from aidevtools.compare.visualizer import Visualizer

        page = Visualizer.create_page(title="My Strategy")

        # 添加图表
        pie = Visualizer.create_pie(...)
        page.add(pie)

        return page
```

### 扩展 Visualizer

在 `visualizer.py` 中添加新的图表类型：

```python
class Visualizer:
    @staticmethod
    def create_scatter(data, title=""):
        """创建散点图"""
        from pyecharts.charts import Scatter
        # ...
```

---

## 6. 关键设计

### 可视化体现策略原理

**BitAnalysis**: 区分 sign/exp/mantissa 错误的语义
**Blocked**: 展示块级分析（局部异常定位）
**Model**: 展示误差传播路径

### 处理缺失数据

**OpStatus.MISSING_DUT**: DUT 未保存算子输出
- Sankey 图用虚线表示
- 饼图统计缺失率
- 可选插值估算

### 精简高效

- 基础底座：纯工具类，无业务逻辑
- 策略级：核心 3-5 个图表
- 模型级：聚焦误差传播和瓶颈定位

---

## 7. 常见问题

**Q: pyecharts 安装失败？**
```bash
pip install --upgrade pip
pip install pyecharts -i https://pypi.tuna.tsinghua.edu.cn/simple
```

**Q: 图表不显示？**
- 检查 HTML 文件路径是否正确
- 确保浏览器可以访问 CDN (echarts.min.js)

**Q: 如何自定义颜色？**
```python
Visualizer.COLORS = {
    'critical': '#FF0000',  # 自定义红色
    'warning': '#FFA500',
    ...
}
```

---

生成时间: 2026-02-08
