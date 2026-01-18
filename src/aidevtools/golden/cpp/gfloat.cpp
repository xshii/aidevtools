/**
 * GFloat Golden API 实现
 */
#include "gfloat.h"
#include <cstring>

namespace gfloat {

// ==================== 指针版本 ====================

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

// ==================== 向量版本 ====================

std::vector<uint16_t> fp32_to_gfloat16(const std::vector<float>& input) {
    std::vector<uint16_t> output(input.size());
    fp32_to_gfloat16(input.data(), input.size(), output.data());
    return output;
}

std::vector<float> gfloat16_to_fp32(const std::vector<uint16_t>& input) {
    std::vector<float> output(input.size());
    gfloat16_to_fp32(input.data(), input.size(), output.data());
    return output;
}

std::vector<uint8_t> fp32_to_gfloat8(const std::vector<float>& input) {
    std::vector<uint8_t> output(input.size());
    fp32_to_gfloat8(input.data(), input.size(), output.data());
    return output;
}

std::vector<float> gfloat8_to_fp32(const std::vector<uint8_t>& input) {
    std::vector<float> output(input.size());
    gfloat8_to_fp32(input.data(), input.size(), output.data());
    return output;
}

}  // namespace gfloat
