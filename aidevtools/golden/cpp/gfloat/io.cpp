/**
 * GFloat I/O 实现
 */
#include "io.h"
#include <cstring>
#include <algorithm>

namespace gfloat_io {

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
    size_t packed_size = gfloat4_packed_size(size);
    std::memset(output, 0, packed_size);

    for (size_t i = 0; i < size; ++i) {
        uint32_t bits;
        std::memcpy(&bits, &input[i], sizeof(float));
        uint8_t val4 = static_cast<uint8_t>(bits >> 28);

        size_t byte_idx = i / 2;
        if (i % 2 == 0) {
            output[byte_idx] |= (val4 << 4);
        } else {
            output[byte_idx] |= val4;
        }
    }
}

void gfloat4_to_fp32(const uint8_t* input, size_t size, float* output) {
    for (size_t i = 0; i < size; ++i) {
        size_t byte_idx = i / 2;
        uint8_t val4;
        if (i % 2 == 0) {
            val4 = (input[byte_idx] >> 4) & 0x0F;
        } else {
            val4 = input[byte_idx] & 0x0F;
        }

        uint32_t bits = static_cast<uint32_t>(val4) << 28;
        std::memcpy(&output[i], &bits, sizeof(float));
    }
}

// ==================== 通用高级接口 ====================

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
            size_t size = packed.size() * 2;
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

}  // namespace gfloat_io
