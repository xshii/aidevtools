# Compare 工具使用指导

## 功能

对比 golden 和仿真器输出，支持 bit/分块/完整级比对，生成报告和热力图。

## 使用

### Python API

```python
from aidevtools.tools.compare import run_compare, archive

# 运行比数
run_compare(
    csv_path="resnet_compare.csv",
    output_dir="./details",
    atol=1e-5,
    rtol=1e-5,
    format="raw",
    dtype=np.float32,
)

# 打包归档
archive("resnet_compare.csv", "resnet_compare.zip")
```

### 命令行

```bash
# 运行比数
aidev compare run resnet_compare.csv

# 只跑单算子
aidev compare run resnet_compare.csv --mode single

# 只跑指定算子
aidev compare run resnet_compare.csv --op conv2d_0

# 打包
aidev compare archive resnet_compare.csv
```

## CSV 格式

| 字段 | 说明 |
|------|------|
| op_name | 算子名 |
| mode | single/chain/full |
| input_bin | 输入文件路径 |
| weight_bin | 权重文件路径 |
| golden_bin | Golden 文件路径 |
| result_bin | 仿真器输出路径 |
| skip | true=跳过 |
| status | PASS/FAIL/SKIP/ERROR |
| max_abs | 最大绝对误差 |
| qsnr | 量化信噪比 (dB) |
| detail_link | 详情目录 |
| note | 备注 |

## 输出

```
resnet_compare/
├── resnet_compare.csv      # 更新后的结果
├── details/
│   ├── conv2d_0/
│   │   ├── summary.txt     # 摘要
│   │   ├── blocks.json     # 分块结果
│   │   ├── heatmap.svg     # 热力图
│   │   └── failed_cases/   # 失败用例
│   └── relu_0/
│       └── ...
└── resnet_compare.zip      # 归档
```

## 精度指标

| 指标 | 说明 | 参考值 |
|------|------|--------|
| max_abs | 最大绝对误差 | < 1e-5 |
| qsnr | 量化信噪比 | > 40dB 优秀 |
| cosine | 余弦相似度 | > 0.999 |
