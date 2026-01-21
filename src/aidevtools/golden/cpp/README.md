# CPU Golden C++ 架构

## 文件结构

```
cpp/
├── main.cpp           # CLI 框架 (参数解析、调度)
├── gfloat_io.h/cpp    # GFloat 格式 I/O
├── bfp_io.h/cpp       # BFP 格式 I/O
├── ops_interface.h    # 算子接口定义
├── ops_impl.cpp       # 算子实现 (可替换)
└── CMakeLists.txt     # 编译配置
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
│   gfloat_io     │       │    bfp_io       │
│  GFloat 格式    │       │   BFP 格式      │
│  I/O + 转换     │       │  I/O + 转换     │
└─────────────────┘       └─────────────────┘
         │                           │
         └─────────────┬─────────────┘
                       ▼
              ┌─────────────────┐
              │  ops_interface  │
              │   算子接口      │
              │  (纯 fp32)      │
              └─────────────────┘
                       │
                       ▼
              ┌─────────────────┐
              │   ops_impl      │
              │   算子实现      │
              │  (可替换)       │
              └─────────────────┘
```

## 如何替换算子实现

1. **保留这些文件**：
   - `main.cpp` (CLI 框架)
   - `gfloat_io.h/cpp` 或 `bfp_io.h/cpp` (格式 I/O)
   - `ops_interface.h` (接口定义)

2. **替换 `ops_impl.cpp`**：
   - 创建你自己的实现文件 (如 `my_ops.cpp`)
   - 实现 `ops_interface.h` 中声明的所有函数
   - 修改 CMakeLists.txt 使用你的文件

3. **修改 CMakeLists.txt**：
   ```cmake
   set(SOURCES
       main.cpp
       gfloat_io.cpp
       my_ops.cpp      # 你的实现
   )
   ```

## 如何使用 BFP 格式

BFP 格式需要同时保存/加载两个文件：
- `mantissa.bin` - 尾数数组
- `exponent.bin` - 共享指数数组

参考 `bfp_io.h` 的接口：

```cpp
#include "bfp_io.h"

using namespace bfp_io;

// 保存
save_as_bfp(fp32_data, size, "mantissa.bin", "exponent.bin", BFPType::BFP8);

// 加载
auto fp32 = load_bfp_as_fp32("mantissa.bin", "exponent.bin", BFPType::BFP8);
```

如果需要 BFP 格式的 CLI，可以：
1. 复制 `main.cpp` 为 `main_bfp.cpp`
2. 修改使用 `bfp_io` 替代 `gfloat_io`
3. 添加新的编译目标

## 算子接口

所有算子都接收 fp32 数据，格式转换在 I/O 层完成：

```cpp
namespace cpu_golden::ops {
    // MatMul: C = A @ B
    void matmul_fp32(const float* a, const float* b, float* c,
                     size_t M, size_t K, size_t N);

    // Softmax
    void softmax_fp32(const float* input, float* output,
                      size_t batch, size_t seq);

    // LayerNorm
    void layernorm_fp32(const float* input, const float* gamma, const float* beta,
                        float* output, size_t batch, size_t hidden, float eps);

    // Transpose 4D
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
