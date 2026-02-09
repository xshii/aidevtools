# 04 xlsx 双向工作流示例

展示 xlsx 配置文件的双向工作流。

## 文件

| 文件 | 说明 |
|------|------|
| mlp_config.xlsx | MLP 模型配置 (4 个算子) |
| run.py | 运行示例 |

## 运行

```bash
cd demos/04_xlsx_basic
python run.py
```

## 两种使用方向

### 方向1: Python → Excel

场景：已有 Python 代码，想导出为 xlsx 配置方便管理

```
Python 代码执行算子
        ↓
    记录 trace
        ↓
导出到 xlsx 配置文件
```

### 方向2: Excel → Python

场景：用 Excel 配置算子用例，自动生成 Python 代码

```
用户编辑 xlsx 配置
        ↓
  生成 Python 代码
        ↓
    运行并比对
```

## xlsx 结构

| Sheet | 说明 |
|-------|------|
| op_registry | 可用算子列表 |
| ops | 算子配置（id, op_name, shape, dtype, depends, qtype, skip, note）|
| compare | 比对结果 |

## 命令行用法

```bash
# 生成空模板
aidev compare xlsx template --output=config.xlsx

# 限定算子的模板
aidev compare xlsx template --output=config.xlsx --ops=linear,relu,attention

# 从 trace 导出到 xlsx
aidev compare xlsx export --xlsx=config.xlsx

# 从 xlsx 生成 Python 代码
aidev compare xlsx import --xlsx=config.xlsx --output=generated.py

# 运行 xlsx 配置并比对
aidev compare xlsx run --xlsx=config.xlsx

# 列出可用算子
aidev compare xlsx ops
```

## 依赖

```bash
pip install openpyxl
```
