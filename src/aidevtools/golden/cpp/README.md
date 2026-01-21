# CPU Golden C++ 架构

## 目录结构

```
cpp/
├── io/                 # I/O 层 (格式转换 + 文件读写)
│   ├── gfloat_io.h
│   ├── gfloat_io.cpp
│   ├── bfp_io.h
│   └── bfp_io.cpp
├── ops/                # 算子层
│   ├── interface.h     # 接口定义 (纯 fp32)
│   └── impl.cpp        # 实现 (可替换)
├── main.cpp            # CLI 框架
├── CMakeLists.txt
└── README.md
```

## 架构设计

```
┌─────────────────────────────────────────────────┐
│                  CLI (main.cpp)                  │
│            参数解析 + 算子调度                    │
└─────────────────────────────────────────────────┘
                       │
         ┌─────────────┴─────────────┐
         ▼                           ▼
┌─────────────────┐       ┌─────────────────┐
│  io/gfloat_io   │       │   io/bfp_io     │
│  GFloat 格式    │       │   BFP 格式      │
│  I/O + 转换     │       │  I/O + 转换     │
└─────────────────┘       └─────────────────┘
         │                           │
         └─────────────┬─────────────┘
                       ▼
              ┌─────────────────┐
              │ ops/interface.h │
              │   算子接口      │
              │  (纯 fp32)      │
              └─────────────────┘
                       │
                       ▼
              ┌─────────────────┐
              │  ops/impl.cpp   │
              │   算子实现      │
              │  (可替换)       │
              └─────────────────┘
```

## 如何替换算子实现

1. **保留这些文件**：
   - `main.cpp` (CLI 框架)
   - `io/gfloat_io.*` 或 `io/bfp_io.*` (格式 I/O)
   - `ops/interface.h` (接口定义)

2. **替换 `ops/impl.cpp`**：
   - 创建你的实现文件 (如 `ops/my_impl.cpp`)
   - 实现 `ops/interface.h` 中声明的所有函数
   - 修改 CMakeLists.txt 使用你的文件

3. **修改 CMakeLists.txt**：
   ```cmake
   set(SOURCES
       main.cpp
       io/gfloat_io.cpp
       ops/my_impl.cpp      # 你的实现
   )
   ```

## 如何使用 BFP 格式

BFP 格式需要同时保存/加载两个文件：
- `mantissa.bin` - 尾数数组
- `exponent.bin` - 共享指数数组

使用方法：
1. 修改 `main.cpp` 的 include：`#include "io/bfp_io.h"`
2. 使用 `bfp_io` namespace 替代 `gfloat_io`
3. 添加 `io/bfp_io.cpp` 到 CMakeLists.txt

## 算子接口

所有算子都接收 fp32 数据，格式转换在 I/O 层完成：

```cpp
namespace cpu_golden::ops {
    void matmul_fp32(const float* a, const float* b, float* c,
                     size_t M, size_t K, size_t N);

    void softmax_fp32(const float* input, float* output,
                      size_t batch, size_t seq);

    void layernorm_fp32(const float* input, const float* gamma, const float* beta,
                        float* output, size_t batch, size_t hidden, float eps);

    void transpose_4d_fp32(const float* input, float* output,
                           size_t d0, size_t d1, size_t d2, size_t d3);
}
```

## 编译

```bash
# 从项目根目录
./build_golden.sh cpu

# 或手动编译
cd src/aidevtools/golden/cpp
mkdir -p build && cd build
cmake ..
make
```

## 测试

```bash
./cpu_golden --help
./cpu_golden matmul gfp16 a.bin b.bin c.bin 64 128 256
```
