# Trace 工具使用指导

## 功能

自动捕获算子的输入输出数据，用于生成 golden 和 compare.csv。

## 使用

```python
from aidevtools import trace, dump, gen_csv

# 1. 给算子加 @trace 装饰器
@trace
def conv2d(x, weight):
    return some_computation(x, weight)

@trace
def relu(x):
    return np.maximum(x, 0)

# 2. 正常运行
x = np.random.randn(1, 3, 224, 224).astype(np.float32)
w = np.random.randn(64, 3, 3, 3).astype(np.float32)

y = conv2d(x, w)
y = relu(y)

# 3. 导出数据
dump("./workspace")

# 4. 生成 compare.csv
gen_csv("./workspace", "resnet")
```

## 输出

```
workspace/
├── conv2d_0_input.bin
├── conv2d_0_weight.bin
├── conv2d_0_golden.bin
├── relu_0_input.bin
├── relu_0_golden.bin
└── resnet_compare.csv
```

## 参数

### @trace 装饰器

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| name | str | 函数名 | 自定义算子名 |
| save_input | bool | True | 是否保存输入 |

### dump()

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| output_dir | str | "./workspace" | 输出目录 |
| format | str | "raw" | 数据格式 |

### gen_csv()

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| output_dir | str | "./workspace" | 输出目录 |
| model_name | str | "model" | 模型名 |
