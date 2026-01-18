# Formats 工具使用指导

## 功能

支持多种数据格式的读写，可扩展。

## 内置格式

| 格式 | 说明 | 扩展名 |
|------|------|--------|
| raw | 纯二进制 | .bin |
| numpy | NumPy 格式 | .npy/.npz |

## 使用

```python
from aidevtools import formats
import numpy as np

data = np.random.randn(1, 3, 224, 224).astype(np.float32)

# Raw 格式
formats.save("data.bin", data, format="raw")
loaded = formats.load("data.bin", format="raw", dtype=np.float32, shape=(1,3,224,224))

# NumPy 格式
formats.save("data.npy", data, format="numpy")
loaded = formats.load("data.npy", format="numpy")
```

## 自定义格式

```python
from aidevtools.formats.base import FormatBase

class MyFormat(FormatBase):
    name = "myformat"

    def load(self, path, **kwargs):
        # 自定义读取逻辑
        ...

    def save(self, path, data, **kwargs):
        # 自定义保存逻辑
        ...

# 自动注册，直接使用
formats.load("data.myfmt", format="myformat")
```
