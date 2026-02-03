/**
 * BFP I/O 实现
 */
#include "io.h"
#include <cstring>

namespace bfp_io {

// ==================== 格式类型解析 ====================

BFPType parse_bfp_type(const std::string& type_str) {
    if (type_str == "bfp4") {
        return BFPType::BFP4;
    } else if (type_str == "bfp8") {
        return BFPType::BFP8;
    } else if (type_str == "bfp16") {
        return BFPType::BFP16;
    }
    throw std::runtime_error("Unknown BFP type: " + type_str);
}

std::string bfp_type_to_string(BFPType type) {
    switch (type) {
        case BFPType::BFP4: return "bfp4";
        case BFPType::BFP8: return "bfp8";
        case BFPType::BFP16: return "bfp16";
        default: return "unknown";
    }
}

int bfp_mantissa_bits(BFPType type) {
    switch (type) {
        case BFPType::BFP4: return 2;
        case BFPType::BFP8: return 4;
        case BFPType::BFP16: return 8;
        default: return 4;
    }
}

int bfp_block_size(BFPType type) {
    switch (type) {
        case BFPType::BFP4: return 64;
        case BFPType::BFP8: return 32;
        case BFPType::BFP16: return 16;
        default: return 32;
    }
}

// ==================== BFP 格式转换 ====================

void fp32_to_bfp(const float* data, size_t size,
                 int block_size, int mantissa_bits,
                 int8_t* mantissas, int8_t* shared_exps) {

    size_t n_blocks = num_blocks(size, block_size);
    int8_t max_mantissa = static_cast<int8_t>((1 << (mantissa_bits - 1)) - 1);

    for (size_t b = 0; b < n_blocks; ++b) {
        size_t start = b * block_size;
        size_t end = std::min(start + static_cast<size_t>(block_size), size);

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
    }
}

void bfp_to_fp32(const int8_t* mantissas, const int8_t* shared_exps,
                 size_t size, int block_size, int mantissa_bits,
                 float* output) {

    size_t n_blocks = num_blocks(size, block_size);

    for (size_t b = 0; b < n_blocks; ++b) {
        size_t start = b * block_size;
        size_t end = std::min(start + static_cast<size_t>(block_size), size);

        int8_t shared_exp = shared_exps[b];

        // 反量化: x = mantissa * 2^(shared_exp - (mantissa_bits - 1))
        float scale = std::pow(2.0f, shared_exp - (mantissa_bits - 1));

        for (size_t i = start; i < end; ++i) {
            output[i] = static_cast<float>(mantissas[i]) * scale;
        }
    }
}

// ==================== 高级接口 ====================

bool save_as_bfp(const float* input, size_t size,
                 const std::string& mantissa_path,
                 const std::string& exponent_path,
                 BFPType type) {

    int block_size = bfp_block_size(type);
    int mantissa_bits = bfp_mantissa_bits(type);
    size_t n_blocks = num_blocks(size, block_size);

    std::vector<int8_t> mantissas(size);
    std::vector<int8_t> shared_exps(n_blocks);

    fp32_to_bfp(input, size, block_size, mantissa_bits,
                mantissas.data(), shared_exps.data());

    bool ok1 = save_binary(mantissa_path, mantissas.data(), size);
    bool ok2 = save_binary(exponent_path, shared_exps.data(), n_blocks);

    return ok1 && ok2;
}

std::vector<float> load_bfp_as_fp32(const std::string& mantissa_path,
                                     const std::string& exponent_path,
                                     BFPType type) {

    std::vector<int8_t> mantissas = load_binary<int8_t>(mantissa_path);
    std::vector<int8_t> shared_exps = load_binary<int8_t>(exponent_path);

    size_t size = mantissas.size();
    int block_size = bfp_block_size(type);
    int mantissa_bits = bfp_mantissa_bits(type);

    std::vector<float> output(size);
    bfp_to_fp32(mantissas.data(), shared_exps.data(),
                size, block_size, mantissa_bits, output.data());

    return output;
}

std::vector<float> load_bfp_as_fp32(const std::string& mantissa_path,
                                     const std::string& exponent_path,
                                     BFPType type,
                                     size_t element_count) {

    std::vector<int8_t> mantissas = load_binary<int8_t>(mantissa_path);
    std::vector<int8_t> shared_exps = load_binary<int8_t>(exponent_path);

    size_t size = std::min(element_count, mantissas.size());
    int block_size = bfp_block_size(type);
    int mantissa_bits = bfp_mantissa_bits(type);

    std::vector<float> output(size);
    bfp_to_fp32(mantissas.data(), shared_exps.data(),
                size, block_size, mantissa_bits, output.data());

    return output;
}

// ==================== 单文件格式接口 ====================

bool save_as_bfp_packed(const float* input, size_t size,
                        const std::string& path, BFPType type) {

    int block_size = bfp_block_size(type);
    int mantissa_bits = bfp_mantissa_bits(type);
    size_t n_blocks = num_blocks(size, block_size);

    std::vector<int8_t> mantissas(size);
    std::vector<int8_t> shared_exps(n_blocks);

    fp32_to_bfp(input, size, block_size, mantissa_bits,
                mantissas.data(), shared_exps.data());

    // 写入单文件: [exps][mantissas]
    std::ofstream file(path, std::ios::binary);
    if (!file.is_open()) {
        return false;
    }
    file.write(reinterpret_cast<const char*>(shared_exps.data()), n_blocks);
    file.write(reinterpret_cast<const char*>(mantissas.data()), size);
    return file.good();
}

std::vector<float> load_bfp_packed_as_fp32(const std::string& path,
                                            BFPType type,
                                            size_t element_count) {

    int block_size = bfp_block_size(type);
    int mantissa_bits = bfp_mantissa_bits(type);
    size_t n_blocks = num_blocks(element_count, block_size);

    // 读取单文件
    std::vector<int8_t> packed = load_binary<int8_t>(path);

    if (packed.size() < n_blocks + element_count) {
        throw std::runtime_error("BFP packed file too small: expected " +
                                  std::to_string(n_blocks + element_count) +
                                  " bytes, got " + std::to_string(packed.size()));
    }

    // 解包: [exps (n_blocks)][mantissas (element_count)]
    const int8_t* shared_exps = packed.data();
    const int8_t* mantissas = packed.data() + n_blocks;

    std::vector<float> output(element_count);
    bfp_to_fp32(mantissas, shared_exps,
                element_count, block_size, mantissa_bits, output.data());

    return output;
}

}  // namespace bfp_io
