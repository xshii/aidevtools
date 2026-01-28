/**
 * 算子实现 (BFP 版本)
 *
 * 这个文件包含所有算子的 BFP 精度模拟实现。
 * BFP (Block Floating Point) 使用块共享指数来减少存储和计算开销。
 *
 * 注意: 这个文件的函数是 BFP 专用版本，通过 bfp_main.cpp 调用。
 * gfloat 版本在 gfloat/impl.cpp 中。
 */
#include "../interface.h"
#include "io.h"
#include <cstring>
#include <cmath>
#include <algorithm>

namespace cpu_golden {
namespace ops {

// BFP 量化辅助宏
#define BFP_QUANTIZE(val, dtype) bfp_io::quantize_fp32_to_bfp_precision(val, dtype)

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

// ==================== BFP 精度模拟版本 ====================
// 以下函数模拟 BFP 硬件的低精度计算行为

void matmul_bfp(const float* a, const float* b, float* c,
                size_t M, size_t K, size_t N, bfp_io::BFPType dtype) {
    std::memset(c, 0, M * N * sizeof(float));

    for (size_t i = 0; i < M; ++i) {
        for (size_t k = 0; k < K; ++k) {
            float a_ik = a[i * K + k];
            for (size_t j = 0; j < N; ++j) {
                float prod = BFP_QUANTIZE(a_ik * b[k * N + j], dtype);
                c[i * N + j] = BFP_QUANTIZE(c[i * N + j] + prod, dtype);
            }
        }
    }
}

void softmax_bfp(const float* input, float* output,
                 size_t batch, size_t seq, bfp_io::BFPType dtype) {
    for (size_t b = 0; b < batch; ++b) {
        const float* row = input + b * seq;
        float* out_row = output + b * seq;

        float max_val = row[0];
        for (size_t i = 1; i < seq; ++i) {
            max_val = std::max(max_val, row[i]);
        }

        float sum = 0.0f;
        for (size_t i = 0; i < seq; ++i) {
            float diff = BFP_QUANTIZE(row[i] - max_val, dtype);
            out_row[i] = BFP_QUANTIZE(std::exp(diff), dtype);
            sum = BFP_QUANTIZE(sum + out_row[i], dtype);
        }

        for (size_t i = 0; i < seq; ++i) {
            out_row[i] = BFP_QUANTIZE(out_row[i] / sum, dtype);
        }
    }
}

void layernorm_bfp(const float* input, const float* gamma, const float* beta,
                   float* output, size_t batch, size_t hidden, float eps, bfp_io::BFPType dtype) {
    for (size_t b = 0; b < batch; ++b) {
        const float* row = input + b * hidden;
        float* out_row = output + b * hidden;

        float mean = 0.0f;
        for (size_t i = 0; i < hidden; ++i) {
            mean = BFP_QUANTIZE(mean + row[i], dtype);
        }
        mean = BFP_QUANTIZE(mean / static_cast<float>(hidden), dtype);

        float var = 0.0f;
        for (size_t i = 0; i < hidden; ++i) {
            float diff = BFP_QUANTIZE(row[i] - mean, dtype);
            var = BFP_QUANTIZE(var + BFP_QUANTIZE(diff * diff, dtype), dtype);
        }
        var = BFP_QUANTIZE(var / static_cast<float>(hidden), dtype);

        float inv_std = BFP_QUANTIZE(1.0f / std::sqrt(BFP_QUANTIZE(var + eps, dtype)), dtype);
        for (size_t i = 0; i < hidden; ++i) {
            float norm = BFP_QUANTIZE(BFP_QUANTIZE(row[i] - mean, dtype) * inv_std, dtype);
            out_row[i] = BFP_QUANTIZE(BFP_QUANTIZE(norm * gamma[i], dtype) + beta[i], dtype);
        }
    }
}

void relu_bfp(const float* input, float* output, size_t size, bfp_io::BFPType dtype) {
    for (size_t i = 0; i < size; ++i) {
        output[i] = BFP_QUANTIZE(std::max(0.0f, input[i]), dtype);
    }
}

void gelu_bfp(const float* input, float* output, size_t size, bfp_io::BFPType dtype) {
    const float sqrt_2_pi = 0.7978845608f;
    const float coef = 0.044715f;

    for (size_t i = 0; i < size; ++i) {
        float x = input[i];
        float x2 = BFP_QUANTIZE(x * x, dtype);
        float x3 = BFP_QUANTIZE(x2 * x, dtype);
        float term1 = BFP_QUANTIZE(coef * x3, dtype);
        float term2 = BFP_QUANTIZE(x + term1, dtype);
        float inner = BFP_QUANTIZE(sqrt_2_pi * term2, dtype);
        float tanh_val = BFP_QUANTIZE(std::tanh(inner), dtype);
        float sum = BFP_QUANTIZE(1.0f + tanh_val, dtype);
        float half_x = BFP_QUANTIZE(0.5f * x, dtype);
        output[i] = BFP_QUANTIZE(half_x * sum, dtype);
    }
}

void sigmoid_bfp(const float* input, float* output, size_t size, bfp_io::BFPType dtype) {
    for (size_t i = 0; i < size; ++i) {
        float neg_x = BFP_QUANTIZE(-input[i], dtype);
        float exp_val = BFP_QUANTIZE(std::exp(neg_x), dtype);
        float denom = BFP_QUANTIZE(1.0f + exp_val, dtype);
        output[i] = BFP_QUANTIZE(1.0f / denom, dtype);
    }
}

void tanh_bfp(const float* input, float* output, size_t size, bfp_io::BFPType dtype) {
    for (size_t i = 0; i < size; ++i) {
        output[i] = BFP_QUANTIZE(std::tanh(input[i]), dtype);
    }
}

void silu_bfp(const float* input, float* output, size_t size, bfp_io::BFPType dtype) {
    for (size_t i = 0; i < size; ++i) {
        float x = input[i];
        float neg_x = BFP_QUANTIZE(-x, dtype);
        float exp_val = BFP_QUANTIZE(std::exp(neg_x), dtype);
        float denom = BFP_QUANTIZE(1.0f + exp_val, dtype);
        float sig = BFP_QUANTIZE(1.0f / denom, dtype);
        output[i] = BFP_QUANTIZE(x * sig, dtype);
    }
}

void add_bfp(const float* a, const float* b, float* c, size_t size, bfp_io::BFPType dtype) {
    for (size_t i = 0; i < size; ++i) {
        c[i] = BFP_QUANTIZE(a[i] + b[i], dtype);
    }
}

void mul_bfp(const float* a, const float* b, float* c, size_t size, bfp_io::BFPType dtype) {
    for (size_t i = 0; i < size; ++i) {
        c[i] = BFP_QUANTIZE(a[i] * b[i], dtype);
    }
}

void div_bfp(const float* a, const float* b, float* c, size_t size, bfp_io::BFPType dtype) {
    for (size_t i = 0; i < size; ++i) {
        c[i] = BFP_QUANTIZE(a[i] / b[i], dtype);
    }
}

#undef BFP_QUANTIZE

}  // namespace ops
}  // namespace cpu_golden
