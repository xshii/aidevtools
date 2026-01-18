# 比数套件使用指导

## 概述

比数套件用于验证自研芯片算子实现的正确性，支持从单算子到完整图的渐进式比对。

## 工作流程

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  1. Trace   │ ──> │  2. 仿真    │ ──> │  3. Compare │
│  生成golden │     │  生成结果   │     │  比对验证   │
└─────────────┘     └─────────────┘     └─────────────┘
      │                                        │
      v                                        v
 compare.csv                            报告 + 归档
```

## 快速开始

### 1. 生成 Golden

```python
from aidevtools import trace, dump, gen_csv

@trace
def conv2d(x, weight):
    # Golden 实现
    return golden_conv2d(x, weight)

# 运行
x = load_input()
w = load_weight()
y = conv2d(x, w)

# 导出
dump("./workspace")
gen_csv("./workspace", "my_model")
```

### 2. 运行仿真器

```bash
# 仿真器生成 result 文件
simulator --input workspace/conv2d_0_input.bin \
          --output workspace/conv2d_0_sim_out.bin
```

### 3. 运行比数

```python
from aidevtools.tools.compare import run_compare

run_compare("my_model_compare.csv")
```

### 4. 查看结果

```
workspace/
├── my_model_compare.csv    # 查看 status 列
└── details/
    └── conv2d_0/
        ├── summary.txt     # 摘要
        └── heatmap.svg     # 热力图
```

## 比对粒度

| 粒度 | 说明 | 用途 |
|------|------|------|
| bit | 二进制完全一致 | 位精确实现 |
| block | 256B 分块对比 | 定位问题区域 |
| full | 完整统计 | 整体评估 |

## 精度指标

| 指标 | 公式 | 参考值 |
|------|------|--------|
| max_abs | max(\|g-r\|) | < 1e-5 |
| qsnr | 10*log10(signal/noise) | > 40dB |
| cosine | dot(g,r)/(norm*norm) | > 0.999 |

## 失败处理

1. 查看 heatmap.svg 定位问题区域
2. 查看 failed_cases/ 获取失败片段
3. 使用片段数据复现问题

## 命令参考

```bash
# 生成 CSV
aidev trace run model.py -o workspace/

# 比数
aidev compare run model_compare.csv
aidev compare run model_compare.csv --mode single
aidev compare run model_compare.csv --op conv2d_0

# 归档
aidev compare archive model_compare.csv
```
