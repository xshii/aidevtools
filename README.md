# AI Dev Tools

AI 算子开发工具集，用于自研芯片算子的 Golden 对比验证。

## 快速上手

```bash
# 1. 安装
./install.sh dev

# 2. 激活环境
source .venv/bin/activate

# 3. 使用
python -c "from aidevtools import trace; print('OK')"
```

## 环境依赖

- Python >= 3.8, < 3.11
- numpy
- pyyaml (可选)

## 功能概览

| 模块 | 功能 |
|------|------|
| trace | 插桩捕获算子输入输出 |
| formats | 多格式数据读写 (raw/numpy/自定义) |
| compare | 比数验证 (bit/block/full) |

## 需求分解

### 核心功能

- [x] 日志模块
- [x] 格式插件 (raw, numpy)
- [x] Trace 插桩
- [x] 比数核心 (diff, report, export)
- [x] CSV 配置表生成
- [x] SVG 热力图
- [x] 失败用例导出
- [x] Zip 归档

### 扩展功能

- [ ] Protobuf 格式
- [ ] 自定义格式示例
- [ ] ONNX 支持
- [ ] CLI 命令集成

### 文档

- [x] 架构设计 (docs/architecture.md)
- [x] 子工具指导 (docs/tools/)
- [x] 比数套件指导 (docs/compare_guide.md)
- [x] 用例正交表 (tests/matrix/)

## 完成情况

| 模块 | 状态 | 说明 |
|------|------|------|
| core/log | ✅ | 日志模块 |
| formats | ✅ | raw + numpy |
| trace | ✅ | 插桩 + 导出 + CSV |
| compare/diff | ✅ | bit/block/full + QSNR |
| compare/report | ✅ | 文本 + SVG |
| compare/export | ✅ | 失败用例导出 |
| compare/archive | ✅ | zip 打包 |
| tests | ✅ | pytest 覆盖 |
| CI | ✅ | ci.sh |

## 目录结构

```
aidevtools/
├── README.md
├── install.sh              # 安装脚本
├── ci.sh                   # 一键 CI
├── docs/
│   ├── architecture.md     # 架构设计
│   ├── compare_guide.md    # 比数套件指导
│   └── tools/              # 子工具指导
├── src/aidevtools/
│   ├── core/               # 核心 (log)
│   ├── formats/            # 数据格式
│   ├── trace/              # 插桩
│   └── tools/compare/      # 比数工具
├── tests/
│   ├── matrix/             # 用例正交表
│   ├── test_formats.py
│   ├── test_trace.py
│   └── test_compare.py
└── libs/prettycli/         # CLI 框架
```

## 使用指导

详见 [docs/compare_guide.md](docs/compare_guide.md)

## 开发

```bash
# 安装开发依赖
./install.sh dev

# 运行测试
pytest tests/ -v

# 一键 CI
./ci.sh
```

## License

MIT
