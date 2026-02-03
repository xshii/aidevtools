/**
 * 算子实现 (默认参考实现)
 *
 * 这个文件包含所有算子的默认实现。
 * 你可以用自己的实现替换这个文件。
 *
 * 替换方法:
 *   1. 保留 ops_interface.h (接口定义)
 *   2. 用你自己的 ops_impl.cpp 替换这个文件
 *   3. 确保实现 ops_interface.h 中声明的所有函数
 *
 * GFloat 精度模拟:
 *   - _gfloat 后缀的函数会在计算过程中量化中间结果
 *   - 这模拟了硬件使用低精度计算的行为
 */
#include "../interface.h"
#include "io.h"
#include <cstring>
#include <cmath>
#include <algorithm>

namespace cpu_golden {
namespace ops {

// 量化辅助宏，用于简化 gfloat 版本的实现
#define Q(val, dtype) gfloat_io::quantize_fp32_to_gfloat_precision(val, dtype)

// ==================== MatMul ====================

void matmul_fp32(const float* a, const float* b, float* c,
                 size_t M, size_t K, size_t N) {
    // 初始化输出为 0
    std::memset(c, 0, M * N * sizeof(float));

    // 矩阵乘法: C[i,j] = sum(A[i,k] * B[k,j])
    // 使用 ikj 循环顺序优化缓存访问
    for (size_t i = 0; i < M; ++i) {
        for (size_t k = 0; k < K; ++k) {
            float a_ik = a[i * K + k];
            for (size_t j = 0; j < N; ++j) {
                c[i * N + j] += a_ik * b[k * N + j];
            }
        }
    }
}

// ==================== Softmax ====================

void softmax_fp32(const float* input, float* output,
                  size_t batch, size_t seq) {
    for (size_t b = 0; b < batch; ++b) {
        const float* row = input + b * seq;
        float* out_row = output + b * seq;

        // 找最大值 (数值稳定性)
        float max_val = row[0];
        for (size_t i = 1; i < seq; ++i) {
            max_val = std::max(max_val, row[i]);
        }

        // exp(x - max) 并求和
        float sum = 0.0f;
        for (size_t i = 0; i < seq; ++i) {
            out_row[i] = std::exp(row[i] - max_val);
            sum += out_row[i];
        }

        // 归一化
        for (size_t i = 0; i < seq; ++i) {
            out_row[i] /= sum;
        }
    }
}

// ==================== LayerNorm ====================

void layernorm_fp32(const float* input, const float* gamma, const float* beta,
                    float* output, size_t batch, size_t hidden, float eps) {
    for (size_t b = 0; b < batch; ++b) {
        const float* row = input + b * hidden;
        float* out_row = output + b * hidden;

        // 计算均值
        float mean = 0.0f;
        for (size_t i = 0; i < hidden; ++i) {
            mean += row[i];
        }
        mean /= static_cast<float>(hidden);

        // 计算方差
        float var = 0.0f;
        for (size_t i = 0; i < hidden; ++i) {
            float diff = row[i] - mean;
            var += diff * diff;
        }
        var /= static_cast<float>(hidden);

        // 归一化: (x - mean) / sqrt(var + eps) * gamma + beta
        float inv_std = 1.0f / std::sqrt(var + eps);
        for (size_t i = 0; i < hidden; ++i) {
            out_row[i] = (row[i] - mean) * inv_std * gamma[i] + beta[i];
        }
    }
}

// ==================== Transpose ====================

void transpose_4d_fp32(const float* input, float* output,
                       size_t d0, size_t d1, size_t d2, size_t d3) {
    // [d0, d1, d2, d3] -> [d0, d1, d3, d2]
    for (size_t i0 = 0; i0 < d0; ++i0) {
        for (size_t i1 = 0; i1 < d1; ++i1) {
            for (size_t i2 = 0; i2 < d2; ++i2) {
                for (size_t i3 = 0; i3 < d3; ++i3) {
                    size_t in_idx = i0 * (d1 * d2 * d3) + i1 * (d2 * d3) + i2 * d3 + i3;
                    size_t out_idx = i0 * (d1 * d3 * d2) + i1 * (d3 * d2) + i3 * d2 + i2;
                    output[out_idx] = input[in_idx];
                }
            }
        }
    }
}

void transpose_2d_fp32(const float* input, float* output, size_t M, size_t N) {
    // [M, N] -> [N, M]
    for (size_t i = 0; i < M; ++i) {
        for (size_t j = 0; j < N; ++j) {
            output[j * M + i] = input[i * N + j];
        }
    }
}

// ==================== 激活函数 ====================

void relu_fp32(const float* input, float* output, size_t size) {
    for (size_t i = 0; i < size; ++i) {
        output[i] = std::max(0.0f, input[i]);
    }
}

void gelu_fp32(const float* input, float* output, size_t size) {
    // GELU: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    const float sqrt_2_pi = 0.7978845608f;  // sqrt(2/pi)
    const float coef = 0.044715f;

    for (size_t i = 0; i < size; ++i) {
        float x = input[i];
        float x3 = x * x * x;
        float inner = sqrt_2_pi * (x + coef * x3);
        output[i] = 0.5f * x * (1.0f + std::tanh(inner));
    }
}

void sigmoid_fp32(const float* input, float* output, size_t size) {
    for (size_t i = 0; i < size; ++i) {
        output[i] = 1.0f / (1.0f + std::exp(-input[i]));
    }
}

void tanh_fp32(const float* input, float* output, size_t size) {
    for (size_t i = 0; i < size; ++i) {
        output[i] = std::tanh(input[i]);
    }
}

void silu_fp32(const float* input, float* output, size_t size) {
    // SiLU: x * sigmoid(x)
    for (size_t i = 0; i < size; ++i) {
        float x = input[i];
        float sig = 1.0f / (1.0f + std::exp(-x));
        output[i] = x * sig;
    }
}

// ==================== 逐元素运算 ====================

void add_fp32(const float* a, const float* b, float* c, size_t size) {
    for (size_t i = 0; i < size; ++i) {
        c[i] = a[i] + b[i];
    }
}

void mul_fp32(const float* a, const float* b, float* c, size_t size) {
    for (size_t i = 0; i < size; ++i) {
        c[i] = a[i] * b[i];
    }
}

void div_fp32(const float* a, const float* b, float* c, size_t size) {
    for (size_t i = 0; i < size; ++i) {
        c[i] = a[i] / b[i];
    }
}

// ==================== GFloat 精度模拟版本 ====================

void matmul_gfloat(const float* a, const float* b, float* c,
                   size_t M, size_t K, size_t N, GFloatType dtype) {
    // 初始化输出为 0
    std::memset(c, 0, M * N * sizeof(float));

    // 矩阵乘法: C[i,j] = sum(A[i,k] * B[k,j])
    // 每次乘加后量化到 gfloat 精度，模拟硬件累加器行为
    for (size_t i = 0; i < M; ++i) {
        for (size_t k = 0; k < K; ++k) {
            float a_ik = a[i * K + k];
            for (size_t j = 0; j < N; ++j) {
                float prod = Q(a_ik * b[k * N + j], dtype);
                c[i * N + j] = Q(c[i * N + j] + prod, dtype);
            }
        }
    }
}

void softmax_gfloat(const float* input, float* output,
                    size_t batch, size_t seq, GFloatType dtype) {
    for (size_t b = 0; b < batch; ++b) {
        const float* row = input + b * seq;
        float* out_row = output + b * seq;

        // 找最大值 (数值稳定性)
        float max_val = row[0];
        for (size_t i = 1; i < seq; ++i) {
            max_val = std::max(max_val, row[i]);
        }

        // exp(x - max) 并求和，每步量化
        float sum = 0.0f;
        for (size_t i = 0; i < seq; ++i) {
            float diff = Q(row[i] - max_val, dtype);
            out_row[i] = Q(std::exp(diff), dtype);
            sum = Q(sum + out_row[i], dtype);
        }

        // 归一化
        for (size_t i = 0; i < seq; ++i) {
            out_row[i] = Q(out_row[i] / sum, dtype);
        }
    }
}

void layernorm_gfloat(const float* input, const float* gamma, const float* beta,
                      float* output, size_t batch, size_t hidden, float eps, GFloatType dtype) {
    for (size_t b = 0; b < batch; ++b) {
        const float* row = input + b * hidden;
        float* out_row = output + b * hidden;

        // 计算均值，每步量化
        float mean = 0.0f;
        for (size_t i = 0; i < hidden; ++i) {
            mean = Q(mean + row[i], dtype);
        }
        mean = Q(mean / static_cast<float>(hidden), dtype);

        // 计算方差，每步量化
        float var = 0.0f;
        for (size_t i = 0; i < hidden; ++i) {
            float diff = Q(row[i] - mean, dtype);
            var = Q(var + Q(diff * diff, dtype), dtype);
        }
        var = Q(var / static_cast<float>(hidden), dtype);

        // 归一化: (x - mean) / sqrt(var + eps) * gamma + beta
        float inv_std = Q(1.0f / std::sqrt(Q(var + eps, dtype)), dtype);
        for (size_t i = 0; i < hidden; ++i) {
            float norm = Q(Q(row[i] - mean, dtype) * inv_std, dtype);
            out_row[i] = Q(Q(norm * gamma[i], dtype) + beta[i], dtype);
        }
    }
}

void relu_gfloat(const float* input, float* output, size_t size, GFloatType dtype) {
    // ReLU 不需要量化中间结果，因为 max(0, x) 只是选择
    // 但输出需要量化以保持一致性
    for (size_t i = 0; i < size; ++i) {
        output[i] = Q(std::max(0.0f, input[i]), dtype);
    }
}

void gelu_gfloat(const float* input, float* output, size_t size, GFloatType dtype) {
    // GELU: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    const float sqrt_2_pi = 0.7978845608f;
    const float coef = 0.044715f;

    for (size_t i = 0; i < size; ++i) {
        float x = input[i];
        float x2 = Q(x * x, dtype);
        float x3 = Q(x2 * x, dtype);
        float term1 = Q(coef * x3, dtype);
        float term2 = Q(x + term1, dtype);
        float inner = Q(sqrt_2_pi * term2, dtype);
        float tanh_val = Q(std::tanh(inner), dtype);
        float sum = Q(1.0f + tanh_val, dtype);
        float half_x = Q(0.5f * x, dtype);
        output[i] = Q(half_x * sum, dtype);
    }
}

void sigmoid_gfloat(const float* input, float* output, size_t size, GFloatType dtype) {
    for (size_t i = 0; i < size; ++i) {
        float neg_x = Q(-input[i], dtype);
        float exp_val = Q(std::exp(neg_x), dtype);
        float denom = Q(1.0f + exp_val, dtype);
        output[i] = Q(1.0f / denom, dtype);
    }
}

void tanh_gfloat(const float* input, float* output, size_t size, GFloatType dtype) {
    for (size_t i = 0; i < size; ++i) {
        output[i] = Q(std::tanh(input[i]), dtype);
    }
}

void silu_gfloat(const float* input, float* output, size_t size, GFloatType dtype) {
    // SiLU: x * sigmoid(x)
    for (size_t i = 0; i < size; ++i) {
        float x = input[i];
        float neg_x = Q(-x, dtype);
        float exp_val = Q(std::exp(neg_x), dtype);
        float denom = Q(1.0f + exp_val, dtype);
        float sig = Q(1.0f / denom, dtype);
        output[i] = Q(x * sig, dtype);
    }
}

void add_gfloat(const float* a, const float* b, float* c, size_t size, GFloatType dtype) {
    for (size_t i = 0; i < size; ++i) {
        c[i] = Q(a[i] + b[i], dtype);
    }
}

void mul_gfloat(const float* a, const float* b, float* c, size_t size, GFloatType dtype) {
    for (size_t i = 0; i < size; ++i) {
        c[i] = Q(a[i] * b[i], dtype);
    }
}

void div_gfloat(const float* a, const float* b, float* c, size_t size, GFloatType dtype) {
    for (size_t i = 0; i < size; ++i) {
        c[i] = Q(a[i] / b[i], dtype);
    }
}

#undef Q

}  // namespace ops
}  // namespace cpu_golden
