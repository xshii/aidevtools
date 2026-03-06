# Compare 策略模式使用指南

> 版本: 3.0
> 日期: 2026-03

## 策略架构

```
CompareEngine
    ↓
CompareStrategy (接口)
    ├── ExactStrategy          # 精确比对 + bit 统计
    ├── FuzzyStrategy          # 模糊比对 (QSNR/cosine)
    ├── SanityStrategy         # Golden 自检
    ├── BlockedStrategy        # 分块定位
    ├── BitAnalysisStrategy    # Bit 级语义分析
    ├── CompositeStrategy      # 自定义组合
    └── TieredStrategy         # 分级策略
        └── ProgressiveStrategy    # 渐进式 (L1→L2→L3)
```

## 引擎工厂

```python
# 渐进式 (默认，早停)
engine = CompareEngine.progressive()

# 深度模式 (三级全执行)
engine = CompareEngine.progressive(deep=True)

# 模型级分析
analyzer = CompareEngine.model_progressive()
```

## 分级说明

### ProgressiveStrategy

| 级别 | 策略 | 作用 | 早停条件 |
|------|------|------|----------|
| L1 | ExactStrategy | 快速筛选 | exact.passed |
| L2 | FuzzyStrategy x2 + SanityStrategy | 中度诊断 | fuzzy.passed |
| L3 | BitAnalysisStrategy + BlockedStrategy | 深度定位 | 最后一级 |

`deep=True` 时所有级别无条件执行。

## 自定义组合

```python
from aidevtools.compare.strategy import CompositeStrategy, ExactStrategy, FuzzyStrategy

engine = CompareEngine(
    strategy=CompositeStrategy([
        ExactStrategy(),
        FuzzyStrategy(use_golden_qnt=False),
    ]),
    config=config,
)
```

## 开发自定义策略

```python
from aidevtools.compare.strategy import CompareStrategy, CompareContext

class MyStrategy(CompareStrategy):
    @property
    def name(self) -> str:
        return "my_custom"

    def run(self, ctx: CompareContext) -> Any:
        # ctx.golden, ctx.dut, ctx.config, ctx.prepared
        return {"passed": True, "score": 0.99}
```

## 按精度选择配置

| 精度 | QSNR | cosine | exceed_ratio |
|------|------|--------|--------------|
| BFP4 | >= 10 dB | >= 0.95 | <= 0.10 |
| BFP8 | >= 15 dB | >= 0.98 | <= 0.05 |
| FP16 | >= 30 dB | >= 0.999 | 0 |
| FP32 | >= 60 dB | >= 0.9999 | 0 |
