# Compare 可视化实现总结

> 实现时间: 2026-02-08
> 状态: ✅ 完成（架构可演进，细节精简高效）

---

## 1. 实现内容

### 1.1 基础底座 (Visualizer)

**文件**: `aidevtools/compare/visualizer.py`

**功能**:
- ✅ 提供 pyecharts 封装（纯工具类）
- ✅ 创建 6 种图表：pie, bar, heatmap, radar, sankey, line
- ✅ 管理颜色方案
- ✅ HTML 渲染

**代码量**: ~150 行

---

### 1.2 策略级可视化

#### BitAnalysisStrategy

**文件**: `aidevtools/compare/strategy/bit_analysis.py`

**新增方法**: `visualize(result) -> Page`

**体现原理**: bit 语义分析
- 错误类型饼图 (Sign Flip / Exponent / Mantissa)
- Bit 布局柱状图 (S|E8|M23)
- 告警摘要 (CRITICAL / WARNING / INFO)

**代码量**: ~60 行

---

#### BlockedStrategy

**文件**: `aidevtools/compare/strategy/blocked.py`

**新增方法**: `visualize(blocks, threshold, cols) -> Page`

**体现原理**: 块级分析（局部 vs 全局）
- QSNR 热力图 (空间分布)
- QSNR 分布直方图 (统计特征)
- 失败块详情 (Top 10)

**代码量**: ~60 行

---

### 1.3 模型级可视化

**文件**: `aidevtools/compare/model_visualizer.py`

**核心类**:
- `OpStatus` - 算子状态 (HAS_DATA / MISSING_DUT / SKIPPED)
- `OpCompareResult` - 单算子结果
- `ModelCompareResult` - 模型级结果
- `ModelVisualizer` - 可视化器

**功能**:
- ✅ 误差传播 Sankey 图
- ✅ 算子 QSNR 排序（找瓶颈）
- ✅ 误差累积曲线
- ✅ 数据完整性饼图（处理 DUT 缺失）

**代码量**: ~200 行

---

### 1.4 示例和文档

**Demo**: `demos/compare/demo_09_visualization.py`
- BitAnalysis 可视化示例
- Blocked 可视化示例
- Model 可视化示例

**文档**:
- `docs/compare_visualization_quickstart.md` - 快速开始
- `docs/compare_visualization_architecture.md` - 架构设计
- `docs/compare_visualization_summary.md` - 本文档

**测试**: `tests/ut/compare/test_visualization.py`
- 基础底座测试
- 策略级测试
- 模型级测试

---

## 2. 架构特点

### 2.1 三层设计

```
┌─────────────────────────────────────┐
│  模型级 (ModelVisualizer)            │
│  - 误差传播分析                      │
│  - 跨算子聚合                        │
│  - 处理 DUT 缺失                     │
└─────────────────────────────────────┘
                ↑
┌─────────────────────────────────────┐
│  策略级 (Strategy.visualize)         │
│  - BitAnalysis: bit 语义分析         │
│  - Blocked: 块级分析                 │
│  - Fuzzy: 统计指标（可扩展）         │
└─────────────────────────────────────┘
                ↑
┌─────────────────────────────────────┐
│  基础底座 (Visualizer)               │
│  - pyecharts 封装                   │
│  - 无业务逻辑                        │
└─────────────────────────────────────┘
```

### 2.2 核心设计思想

**可视化体现策略原理**:
- BitAnalysis: 展示 sign/exp/mantissa 错误的语义差异
- Blocked: 展示块级分析（空间分布 + 统计特征）
- Model: 展示误差传播路径和瓶颈定位

**处理现实约束**:
- DUT 不保存所有算子输出 → `OpStatus.MISSING_DUT`
- 数据完整性可视化
- Sankey 图标注缺失节点

**精简高效**:
- 基础底座：150 行
- 策略级：60 行/策略
- 模型级：200 行
- **总计**: ~470 行核心代码

---

## 3. 使用示例

### 3.1 BitAnalysis 可视化

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

### 3.2 Blocked 可视化

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

### 3.3 Model 可视化

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
    OpCompareResult("relu1", 1, OpStatus.MISSING_DUT),  # DUT 未保存
    OpCompareResult("conv2", 2, OpStatus.HAS_DATA, qsnr=12.3, passed=False),  # 瓶颈
]

model_result = ModelCompareResult(
    model_name="ResNet50",
    ops=ops,
    total_ops=3,
    ops_with_data=2,
    ops_missing_dut=1,
    passed_ops=1,
    failed_ops=1,
)

# 可视化
page = ModelVisualizer.visualize(model_result)
page.render("model_report.html")
```

---

## 4. 扩展点

### 4.1 添加新策略的可视化

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

### 4.2 扩展 Visualizer

```python
class Visualizer:
    @staticmethod
    def create_scatter(data, title=""):
        """创建散点图"""
        from pyecharts.charts import Scatter
        # ...
```

### 4.3 扩展 ModelVisualizer

```python
class ModelVisualizer:
    @staticmethod
    def create_tradeoff_chart(result):
        """Pareto 前沿（QSNR vs Latency）"""
        # ...
```

---

## 5. 依赖

**必需**:
- `pyecharts` (pip install pyecharts)

**可选**:
- `matplotlib` (如果需要额外的图表类型)

---

## 6. 测试

```bash
# 安装依赖
pip install pyecharts

# 运行测试
pytest tests/ut/compare/test_visualization.py

# 运行 demo
python demos/compare/demo_09_visualization.py

# 输出:
# ✅ HTML report saved: /tmp/bitwise_report.html
# ✅ HTML report saved: /tmp/blocked_report.html
# ✅ HTML report saved: /tmp/model_report.html
```

---

## 7. 与业界对比

| 功能 | TensorBoard | Weights & Biases | **aidevtools (实现)** |
|------|-------------|-----------------|---------------------|
| 交互式图表 | ✅ | ✅ | ✅ (pyecharts) |
| 多策略对比 | ❌ | ✅ | ✅ (策略级) |
| 误差传播分析 | ❌ | ❌ | ✅ (模型级) |
| 处理缺失数据 | ❌ | ❌ | ✅ (OpStatus) |
| 代码量 | 大 | - | **470 行** |
| 依赖 | 重 | 云端 | **轻** (pyecharts) |

---

## 8. 下一步

### 8.1 短期（已完成）

- ✅ 基础底座 (Visualizer)
- ✅ BitAnalysis 可视化
- ✅ Blocked 可视化
- ✅ ModelVisualizer
- ✅ Demo 和文档

### 8.2 中期（可选）

- [ ] FuzzyStrategy 可视化（雷达图）
- [ ] ExactStrategy 可视化（仪表盘）
- [ ] EngineVisualizer（综合报告）

### 8.3 长期（可选）

- [ ] 实验追踪数据库（SQLite）
- [ ] 多次运行对比
- [ ] Pareto 前沿分析
- [ ] Jupyter Notebook 集成

---

## 9. 总结

### 9.1 实现成果

✅ **三层架构**：清晰分层，职责明确
✅ **体现原理**：可视化反映策略设计思想
✅ **精简高效**：470 行核心代码
✅ **可扩展**：预留清晰的扩展点
✅ **处理约束**：DUT 缺失数据

### 9.2 关键特点

1. **pyecharts 驱动** - 交互式图表，无需 matplotlib
2. **策略原理可视化** - 不是简单画图
3. **误差传播分析** - 模型级跨算子聚合
4. **处理缺失数据** - OpStatus 标注
5. **单文件 HTML** - 易于分享

### 9.3 工作量

- **基础底座**: 1.5h
- **BitAnalysis**: 1h
- **Blocked**: 1h
- **ModelVisualizer**: 2h
- **Demo 和文档**: 0.5h
- **总计**: **6h**（精简高效）

---

生成时间: 2026-02-08
实现状态: ✅ 完成
