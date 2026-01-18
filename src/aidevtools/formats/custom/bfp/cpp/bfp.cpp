/**
 * Block Floating Point (BFP) Golden Implementation
 */
#include "bfp.h"
#include <cmath>
#include <algorithm>
#include <limits>

namespace bfp {

void fp32_to_bfp(const float* data, size_t size,
                 int block_size, int mantissa_bits,
                 int8_t* mantissas, int8_t* shared_exps) {

    size_t n_blocks = num_blocks(size, block_size);
    int8_t max_mantissa = (1 << (mantissa_bits - 1)) - 1;

    for (size_t b = 0; b < n_blocks; ++b) {
        size_t start = b * block_size;
        size_t end = std::min(start + block_size, size);

        // 1. 找块内最大绝对值
        float max_abs = 0.0f;
        for (size_t i = start; i < end; ++i) {
            float abs_val = std::fabs(data[i]);
            if (abs_val > max_abs) {
                max_abs = abs_val;
            }
        }

        // 2. 计算共享指数
        int8_t shared_exp;
        if (max_abs < 1e-10f) {
            shared_exp = -127;  // 特殊值表示全零块
        } else {
            shared_exp = static_cast<int8_t>(std::floor(std::log2(max_abs))) + 1;
        }
        shared_exps[b] = shared_exp;

        // 3. 量化每个元素
        float scale = std::pow(2.0f, mantissa_bits - 1 - shared_exp);
        for (size_t i = start; i < end; ++i) {
            float scaled = data[i] * scale;
            int rounded = static_cast<int>(std::round(scaled));
            // 裁剪
            rounded = std::max(-static_cast<int>(max_mantissa),
                              std::min(static_cast<int>(max_mantissa), rounded));
            mantissas[i] = static_cast<int8_t>(rounded);
        }

        // 4. 填充块尾部（如果需要）
        for (size_t i = end; i < start + block_size && i < size + (block_size - size % block_size) % block_size; ++i) {
            if (i < size + block_size) {
                // 只在有效范围内填充
            }
        }
    }
}

void bfp_to_fp32(const int8_t* mantissas, const int8_t* shared_exps,
                 size_t size, int block_size, int mantissa_bits,
                 float* output) {

    size_t n_blocks = num_blocks(size, block_size);

    for (size_t b = 0; b < n_blocks; ++b) {
        size_t start = b * block_size;
        size_t end = std::min(start + block_size, size);

        int8_t shared_exp = shared_exps[b];

        // 反量化: x = mantissa * 2^(shared_exp - (mantissa_bits - 1))
        float scale = std::pow(2.0f, shared_exp - (mantissa_bits - 1));

        for (size_t i = start; i < end; ++i) {
            output[i] = static_cast<float>(mantissas[i]) * scale;
        }
    }
}

} // namespace bfp
