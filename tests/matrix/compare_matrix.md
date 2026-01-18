# 比数用例正交表

## 维度

| 维度 | 取值 |
|------|------|
| 数据格式 | raw, numpy, proto |
| 数据类型 | float32, float16, int8 |
| 对比粒度 | bit, block, full |
| 数据规模 | 小(1KB), 中(1MB), 大(100MB) |
| 误差类型 | 无误差, 小误差, 大误差 |
| 模式 | single, chain, full |

## 用例矩阵

| ID | 格式 | dtype | 粒度 | 规模 | 误差 | 模式 | 预期 |
|----|------|-------|------|------|------|------|------|
| C001 | raw | float32 | full | 小 | 无 | single | PASS |
| C002 | raw | float32 | full | 小 | 小 | single | PASS |
| C003 | raw | float32 | full | 小 | 大 | single | FAIL |
| C004 | raw | float32 | block | 中 | 小 | single | PASS |
| C005 | raw | float32 | block | 中 | 大 | single | FAIL |
| C006 | raw | float32 | bit | 小 | 无 | single | PASS |
| C007 | raw | float32 | bit | 小 | 小 | single | FAIL |
| C008 | numpy | float32 | full | 中 | 小 | single | PASS |
| C009 | raw | float16 | full | 小 | 小 | single | PASS |
| C010 | raw | int8 | full | 小 | 无 | single | PASS |
| C011 | raw | float32 | full | 大 | 小 | single | PASS |
| C012 | raw | float32 | full | 小 | 小 | chain | PASS |
| C013 | raw | float32 | full | 中 | 小 | full | PASS |

## 边界用例

| ID | 场景 | 预期 |
|----|------|------|
| B001 | 空数据 | ERROR |
| B002 | shape 不匹配 | ERROR |
| B003 | dtype 不匹配 | ERROR |
| B004 | 文件不存在 | SKIP |
| B005 | 全零数据 | PASS |
| B006 | 包含 NaN | FAIL |
| B007 | 包含 Inf | FAIL |
