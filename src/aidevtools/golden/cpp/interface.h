/**
 * 算子接口定义
 *
 * 这个文件定义了所有算子的纯接口。
 * 所有函数都接收 fp32 格式的数据，格式转换在 I/O 层完成。
 *
 * 你可以:
 *   1. 使用默认的 ops_impl.cpp 实现
 *   2. 替换为你自己的实现 (只需实现这些函数)
 *
 * 数据格式:
 *   - 所有输入输出都是 fp32 (float*)
 *   - 矩阵以行优先 (row-major) 存储
 *   - 批次维度在最前面
 *
 * 精度模拟:
 *   - 带 _gfloat/_bfp 后缀的函数会在计算过程中模拟硬件精度
 *   - 中间结果会被量化到指定的精度
 *   - 这用于模拟 gfp4/gfp8/gfp16 或 bfp4/bfp8/bfp16 硬件行为
 */
#pragma once

#include <cstddef>

// 前向声明格式类型（避免直接 include 格式特定头文件）
namespace gfloat_io { enum class GFloatType; }
namespace bfp_io { enum class BFPType; }

using gfloat_io::GFloatType;
using bfp_io::BFPType;

namespace cpu_golden {
namespace ops {

// ==================== MatMul ====================

/**
 * 矩阵乘法: C = A @ B
 *
 * @param a     输入矩阵 A, shape: [M, K], 行优先
 * @param b     输入矩阵 B, shape: [K, N], 行优先
 * @param c     输出矩阵 C, shape: [M, N], 行优先
 * @param M     A 的行数
 * @param K     A 的列数 / B 的行数
 * @param N     B 的列数
 */
void matmul_fp32(const float* a, const float* b, float* c,
                 size_t M, size_t K, size_t N);

// ==================== Softmax ====================

/**
 * Softmax: y = softmax(x, axis=-1)
 *
 * @param input   输入数据, shape: [batch, seq]
 * @param output  输出数据, shape: [batch, seq]
 * @param batch   批次大小
 * @param seq     序列长度 (softmax 作用的维度)
 */
void softmax_fp32(const float* input, float* output,
                  size_t batch, size_t seq);

// ==================== LayerNorm ====================

/**
 * Layer Normalization: y = (x - mean) / sqrt(var + eps) * gamma + beta
 *
 * @param input   输入数据, shape: [batch, hidden]
 * @param gamma   缩放参数, shape: [hidden]
 * @param beta    偏置参数, shape: [hidden]
 * @param output  输出数据, shape: [batch, hidden]
 * @param batch   批次大小
 * @param hidden  隐藏层大小 (归一化作用的维度)
 * @param eps     数值稳定性参数, 默认 1e-5
 */
void layernorm_fp32(const float* input, const float* gamma, const float* beta,
                    float* output, size_t batch, size_t hidden, float eps = 1e-5f);

// ==================== Transpose ====================

/**
 * 4D 转置: 交换最后两个维度
 * [d0, d1, d2, d3] -> [d0, d1, d3, d2]
 *
 * @param input   输入数据
 * @param output  输出数据
 * @param d0, d1, d2, d3  输入形状
 */
void transpose_4d_fp32(const float* input, float* output,
                       size_t d0, size_t d1, size_t d2, size_t d3);

/**
 * 2D 转置: [M, N] -> [N, M]
 *
 * @param input   输入数据, shape: [M, N]
 * @param output  输出数据, shape: [N, M]
 * @param M       输入行数
 * @param N       输入列数
 */
void transpose_2d_fp32(const float* input, float* output, size_t M, size_t N);

// ==================== 激活函数 ====================

/**
 * ReLU: y = max(0, x)
 *
 * @param input   输入数据
 * @param output  输出数据
 * @param size    元素数量
 */
void relu_fp32(const float* input, float* output, size_t size);

/**
 * GELU: y = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
 *
 * @param input   输入数据
 * @param output  输出数据
 * @param size    元素数量
 */
void gelu_fp32(const float* input, float* output, size_t size);

/**
 * Sigmoid: y = 1 / (1 + exp(-x))
 *
 * @param input   输入数据
 * @param output  输出数据
 * @param size    元素数量
 */
void sigmoid_fp32(const float* input, float* output, size_t size);

/**
 * Tanh: y = tanh(x)
 *
 * @param input   输入数据
 * @param output  输出数据
 * @param size    元素数量
 */
void tanh_fp32(const float* input, float* output, size_t size);

/**
 * SiLU/Swish: y = x * sigmoid(x)
 *
 * @param input   输入数据
 * @param output  输出数据
 * @param size    元素数量
 */
void silu_fp32(const float* input, float* output, size_t size);

// ==================== 逐元素运算 ====================

/**
 * Add: c = a + b
 *
 * @param a, b    输入数据
 * @param c       输出数据
 * @param size    元素数量
 */
void add_fp32(const float* a, const float* b, float* c, size_t size);

/**
 * Mul: c = a * b
 *
 * @param a, b    输入数据
 * @param c       输出数据
 * @param size    元素数量
 */
void mul_fp32(const float* a, const float* b, float* c, size_t size);

/**
 * Div: c = a / b
 *
 * @param a, b    输入数据
 * @param c       输出数据
 * @param size    元素数量
 */
void div_fp32(const float* a, const float* b, float* c, size_t size);

// ==================== GFloat 精度模拟版本 ====================
// 以下函数在计算过程中模拟硬件的低精度行为
// 每个中间结果都会被量化到指定的 gfloat 精度

/**
 * MatMul (gfloat 精度): 中间累加使用 gfloat 精度
 */
void matmul_gfloat(const float* a, const float* b, float* c,
                   size_t M, size_t K, size_t N, GFloatType dtype);

/**
 * Softmax (gfloat 精度): exp/sum 中间结果使用 gfloat 精度
 */
void softmax_gfloat(const float* input, float* output,
                    size_t batch, size_t seq, GFloatType dtype);

/**
 * LayerNorm (gfloat 精度): mean/var 计算使用 gfloat 精度
 */
void layernorm_gfloat(const float* input, const float* gamma, const float* beta,
                      float* output, size_t batch, size_t hidden, float eps, GFloatType dtype);

/**
 * ReLU (gfloat 精度)
 */
void relu_gfloat(const float* input, float* output, size_t size, GFloatType dtype);

/**
 * GELU (gfloat 精度): 中间结果使用 gfloat 精度
 */
void gelu_gfloat(const float* input, float* output, size_t size, GFloatType dtype);

/**
 * Sigmoid (gfloat 精度)
 */
void sigmoid_gfloat(const float* input, float* output, size_t size, GFloatType dtype);

/**
 * Tanh (gfloat 精度)
 */
void tanh_gfloat(const float* input, float* output, size_t size, GFloatType dtype);

/**
 * SiLU (gfloat 精度)
 */
void silu_gfloat(const float* input, float* output, size_t size, GFloatType dtype);

/**
 * Add (gfloat 精度)
 */
void add_gfloat(const float* a, const float* b, float* c, size_t size, GFloatType dtype);

/**
 * Mul (gfloat 精度)
 */
void mul_gfloat(const float* a, const float* b, float* c, size_t size, GFloatType dtype);

/**
 * Div (gfloat 精度)
 */
void div_gfloat(const float* a, const float* b, float* c, size_t size, GFloatType dtype);

// ==================== BFP 精度模拟版本 ====================
// 以下函数在计算过程中模拟 BFP 硬件的低精度行为
// 每个中间结果都会被量化到指定的 BFP 精度

/**
 * MatMul (BFP 精度): 中间累加使用 BFP 精度
 */
void matmul_bfp(const float* a, const float* b, float* c,
                size_t M, size_t K, size_t N, BFPType dtype);

/**
 * Softmax (BFP 精度)
 */
void softmax_bfp(const float* input, float* output,
                 size_t batch, size_t seq, BFPType dtype);

/**
 * LayerNorm (BFP 精度)
 */
void layernorm_bfp(const float* input, const float* gamma, const float* beta,
                   float* output, size_t batch, size_t hidden, float eps, BFPType dtype);

/**
 * ReLU (BFP 精度)
 */
void relu_bfp(const float* input, float* output, size_t size, BFPType dtype);

/**
 * GELU (BFP 精度)
 */
void gelu_bfp(const float* input, float* output, size_t size, BFPType dtype);

/**
 * Sigmoid (BFP 精度)
 */
void sigmoid_bfp(const float* input, float* output, size_t size, BFPType dtype);

/**
 * Tanh (BFP 精度)
 */
void tanh_bfp(const float* input, float* output, size_t size, BFPType dtype);

/**
 * SiLU (BFP 精度)
 */
void silu_bfp(const float* input, float* output, size_t size, BFPType dtype);

/**
 * Add (BFP 精度)
 */
void add_bfp(const float* a, const float* b, float* c, size_t size, BFPType dtype);

/**
 * Mul (BFP 精度)
 */
void mul_bfp(const float* a, const float* b, float* c, size_t size, BFPType dtype);

/**
 * Div (BFP 精度)
 */
void div_bfp(const float* a, const float* b, float* c, size_t size, BFPType dtype);

}  // namespace ops
}  // namespace cpu_golden
