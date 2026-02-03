# CPU Golden C++ 架构

## 目录结构

```
cpp/
├── gfloat/           # GFloat 格式全套
│   ├── io.h          # I/O + 格式转换
│   ├── io.cpp
│   └── impl.cpp      # 算子实现 (可替换)
├── bfp/              # BFP 格式全套
│   ├── io.h          # I/O + 格式转换
│   ├── io.cpp
│   └── impl.cpp      # 算子实现 (可替换)
├── interface.h       # 通用算子接口 (纯 fp32)
├── main.cpp          # CLI 框架
└── CMakeLists.txt
```

## 架构设计

每个格式是独立的模块，包含 I/O 和算子实现：

```
┌─────────────────────────────────────────────────┐
│                  CLI (main.cpp)                  │
│            参数解析 + 算子调度                    │
└─────────────────────────────────────────────────┘
                       │
         ┌─────────────┴─────────────┐
         ▼                           ▼
┌─────────────────┐       ┌─────────────────┐
│    gfloat/      │       │     bfp/        │
│  ├── io.h/cpp   │       │  ├── io.h/cpp   │
│  └── impl.cpp   │       │  └── impl.cpp   │
└─────────────────┘       └─────────────────┘
         │                           │
         └─────────────┬─────────────┘
                       ▼
              ┌─────────────────┐
              │  interface.h    │
              │   算子接口      │
              │  (纯 fp32)      │
              └─────────────────┘
```

## 切换数据格式

修改 CMakeLists.txt 中的一行：

```cmake
# 使用 GFloat 格式
set(FORMAT "gfloat")

# 或使用 BFP 格式
set(FORMAT "bfp")
```

## 替换算子实现

只需替换对应格式目录下的 `impl.cpp`：

```bash
# 替换 GFloat 格式的算子实现
cp my_gfloat_impl.cpp gfloat/impl.cpp

# 或替换 BFP 格式的算子实现
cp my_bfp_impl.cpp bfp/impl.cpp
```

## 算子接口

所有算子都接收 fp32 数据 (`interface.h`)：

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
rm -rf build && mkdir build && cd build
cmake ..
make
```
