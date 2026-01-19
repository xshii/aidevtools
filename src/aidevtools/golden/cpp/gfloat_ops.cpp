/**
 * GFloat CPU Golden Ops 实现
 *
 * 支持 gfloat4/8/16 三种格式
 */
#include "gfloat_ops.h"
#include <cstring>
#include <cmath>
#include <algorithm>

namespace gfloat_ops {

// ==================== 格式类型解析 ====================

GFloatType parse_gfloat_type(const std::string& type_str) {
    if (type_str == "gfloat4" || type_str == "gfp4" || type_str == "4") {
        return GFloatType::GFLOAT4;
    } else if (type_str == "gfloat8" || type_str == "gfp8" || type_str == "8") {
        return GFloatType::GFLOAT8;
    } else if (type_str == "gfloat16" || type_str == "gfp16" || type_str == "16") {
        return GFloatType::GFLOAT16;
    }
    throw std::runtime_error("Unknown gfloat type: " + type_str);
}

std::string gfloat_type_to_string(GFloatType type) {
    switch (type) {
        case GFloatType::GFLOAT4: return "gfloat4";
        case GFloatType::GFLOAT8: return "gfloat8";
        case GFloatType::GFLOAT16: return "gfloat16";
        default: return "unknown";
    }
}

// ==================== GFloat16 格式转换 ====================

void fp32_to_gfloat16(const float* input, size_t size, uint16_t* output) {
    for (size_t i = 0; i < size; ++i) {
        uint32_t bits;
        std::memcpy(&bits, &input[i], sizeof(float));
        output[i] = static_cast<uint16_t>(bits >> 16);
    }
}

void gfloat16_to_fp32(const uint16_t* input, size_t size, float* output) {
    for (size_t i = 0; i < size; ++i) {
        uint32_t bits = static_cast<uint32_t>(input[i]) << 16;
        std::memcpy(&output[i], &bits, sizeof(float));
    }
}

// ==================== GFloat8 格式转换 ====================

void fp32_to_gfloat8(const float* input, size_t size, uint8_t* output) {
    for (size_t i = 0; i < size; ++i) {
        uint32_t bits;
        std::memcpy(&bits, &input[i], sizeof(float));
        output[i] = static_cast<uint8_t>(bits >> 24);
    }
}

void gfloat8_to_fp32(const uint8_t* input, size_t size, float* output) {
    for (size_t i = 0; i < size; ++i) {
        uint32_t bits = static_cast<uint32_t>(input[i]) << 24;
        std::memcpy(&output[i], &bits, sizeof(float));
    }
}

// ==================== GFloat4 格式转换 (packed) ====================

void fp32_to_gfloat4(const float* input, size_t size, uint8_t* output) {
    // 2 个 4-bit 值打包到 1 个 uint8_t
    // 高 4 位存偶数索引，低 4 位存奇数索引
    size_t packed_size = gfloat4_packed_size(size);
    std::memset(output, 0, packed_size);

    for (size_t i = 0; i < size; ++i) {
        uint32_t bits;
        std::memcpy(&bits, &input[i], sizeof(float));
        uint8_t val4 = static_cast<uint8_t>(bits >> 28);  // 取高 4 位

        size_t byte_idx = i / 2;
        if (i % 2 == 0) {
            output[byte_idx] |= (val4 << 4);  // 高 4 位
        } else {
            output[byte_idx] |= val4;          // 低 4 位
        }
    }
}

void gfloat4_to_fp32(const uint8_t* input, size_t size, float* output) {
    // size 是元素数量，不是字节数
    for (size_t i = 0; i < size; ++i) {
        size_t byte_idx = i / 2;
        uint8_t val4;
        if (i % 2 == 0) {
            val4 = (input[byte_idx] >> 4) & 0x0F;  // 高 4 位
        } else {
            val4 = input[byte_idx] & 0x0F;         // 低 4 位
        }

        uint32_t bits = static_cast<uint32_t>(val4) << 28;
        std::memcpy(&output[i], &bits, sizeof(float));
    }
}

// ==================== 通用接口 ====================

bool save_as_gfloat(const float* input, size_t size, const std::string& path, GFloatType type) {
    switch (type) {
        case GFloatType::GFLOAT4: {
            std::vector<uint8_t> packed(gfloat4_packed_size(size));
            fp32_to_gfloat4(input, size, packed.data());
            return save_gfloat4_packed(path, packed.data(), packed.size());
        }
        case GFloatType::GFLOAT8: {
            std::vector<uint8_t> data(size);
            fp32_to_gfloat8(input, size, data.data());
            return save_gfloat8(path, data.data(), size);
        }
        case GFloatType::GFLOAT16: {
            std::vector<uint16_t> data(size);
            fp32_to_gfloat16(input, size, data.data());
            return save_gfloat16(path, data.data(), size);
        }
        default:
            return false;
    }
}

std::vector<float> load_gfloat_as_fp32(const std::string& path, GFloatType type) {
    switch (type) {
        case GFloatType::GFLOAT4: {
            std::vector<uint8_t> packed = load_gfloat4_packed(path);
            size_t size = packed.size() * 2;  // 每个字节 2 个元素
            std::vector<float> output(size);
            gfloat4_to_fp32(packed.data(), size, output.data());
            return output;
        }
        case GFloatType::GFLOAT8: {
            std::vector<uint8_t> data = load_gfloat8(path);
            std::vector<float> output(data.size());
            gfloat8_to_fp32(data.data(), data.size(), output.data());
            return output;
        }
        case GFloatType::GFLOAT16: {
            std::vector<uint16_t> data = load_gfloat16(path);
            std::vector<float> output(data.size());
            gfloat16_to_fp32(data.data(), data.size(), output.data());
            return output;
        }
        default:
            return std::vector<float>();
    }
}

std::vector<float> load_gfloat_as_fp32(const std::string& path, GFloatType type, size_t element_count) {
    switch (type) {
        case GFloatType::GFLOAT4: {
            std::vector<uint8_t> packed = load_gfloat4_packed(path);
            std::vector<float> output(element_count);
            gfloat4_to_fp32(packed.data(), element_count, output.data());
            return output;
        }
        case GFloatType::GFLOAT8: {
            std::vector<uint8_t> data = load_gfloat8(path);
            size_t actual_count = std::min(element_count, data.size());
            std::vector<float> output(actual_count);
            gfloat8_to_fp32(data.data(), actual_count, output.data());
            return output;
        }
        case GFloatType::GFLOAT16: {
            std::vector<uint16_t> data = load_gfloat16(path);
            size_t actual_count = std::min(element_count, data.size());
            std::vector<float> output(actual_count);
            gfloat16_to_fp32(data.data(), actual_count, output.data());
            return output;
        }
        default:
            return std::vector<float>();
    }
}

// ==================== MatMul ====================

void matmul_fp32(const float* a, const float* b, float* c,
                 size_t M, size_t K, size_t N) {
    // 初始化输出为 0
    std::memset(c, 0, M * N * sizeof(float));

    // 矩阵乘法: C[i,j] = sum(A[i,k] * B[k,j])
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
    // 输入 stride: [d1*d2*d3, d2*d3, d3, 1]
    // 输出 stride: [d1*d3*d2, d3*d2, 1, d2]
    for (size_t i0 = 0; i0 < d0; ++i0) {
        for (size_t i1 = 0; i1 < d1; ++i1) {
            for (size_t i2 = 0; i2 < d2; ++i2) {
                for (size_t i3 = 0; i3 < d3; ++i3) {
                    // 输入索引: [i0, i1, i2, i3]
                    size_t in_idx = i0 * (d1 * d2 * d3) + i1 * (d2 * d3) + i2 * d3 + i3;
                    // 输出索引: [i0, i1, i3, i2] (交换最后两维)
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

}  // namespace gfloat_ops
