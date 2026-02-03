/**
 * BFP I/O - Block Floating Point 格式文件读写与转换
 *
 * BFP 格式说明:
 *   - 数据分块，每块共享一个指数
 *   - 每个元素只存尾数 (mantissa)
 *   - 支持 bfp4 (2-bit mantissa), bfp8 (4-bit), bfp16 (8-bit)
 *
 * 文件格式:
 *   - mantissa 文件: int8_t 数组，每个元素一个 mantissa
 *   - exponent 文件: int8_t 数组，每个块一个 shared exponent
 */
#pragma once

#include <cstdint>
#include <cstddef>
#include <vector>
#include <string>
#include <fstream>
#include <stdexcept>
#include <cmath>
#include <algorithm>

namespace bfp_io {

// ==================== BFP 格式类型 ====================

enum class BFPType {
    BFP4,   // 2-bit mantissa, block_size=64
    BFP8,   // 4-bit mantissa, block_size=32
    BFP16   // 8-bit mantissa, block_size=16
};

/**
 * 根据字符串解析格式类型
 * 支持: "bfp4", "bfp8", "bfp16"
 */
BFPType parse_bfp_type(const std::string& type_str);

/**
 * 格式类型转字符串
 */
std::string bfp_type_to_string(BFPType type);

/**
 * 获取格式的 mantissa 位数
 */
int bfp_mantissa_bits(BFPType type);

/**
 * 获取格式的 block size
 */
int bfp_block_size(BFPType type);

// ==================== 基础文件 I/O ====================

/**
 * 从 binary 文件加载数据
 */
template<typename T>
std::vector<T> load_binary(const std::string& path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file: " + path);
    }

    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);

    size_t count = size / sizeof(T);
    std::vector<T> data(count);

    if (!file.read(reinterpret_cast<char*>(data.data()), size)) {
        throw std::runtime_error("Failed to read file: " + path);
    }

    return data;
}

/**
 * 保存数据到 binary 文件
 */
template<typename T>
bool save_binary(const std::string& path, const T* data, size_t count) {
    std::ofstream file(path, std::ios::binary);
    if (!file.is_open()) {
        return false;
    }
    file.write(reinterpret_cast<const char*>(data), count * sizeof(T));
    return file.good();
}

// 便捷别名
inline std::vector<float> load_fp32(const std::string& path) {
    return load_binary<float>(path);
}

inline bool save_fp32(const std::string& path, const float* data, size_t size) {
    return save_binary(path, data, size);
}

// ==================== BFP 格式转换 ====================

/**
 * 计算块数
 */
inline size_t num_blocks(size_t size, int block_size) {
    return (size + block_size - 1) / block_size;
}

/**
 * fp32 -> BFP 量化
 *
 * @param data 输入 fp32 数据
 * @param size 数据大小
 * @param block_size 块大小
 * @param mantissa_bits 尾数位数
 * @param mantissas 输出尾数数组 (size 个元素)
 * @param shared_exps 输出共享指数数组 (num_blocks 个元素)
 */
void fp32_to_bfp(const float* data, size_t size,
                 int block_size, int mantissa_bits,
                 int8_t* mantissas, int8_t* shared_exps);

/**
 * BFP -> fp32 反量化
 *
 * @param mantissas 尾数数组
 * @param shared_exps 共享指数数组
 * @param size 数据大小
 * @param block_size 块大小
 * @param mantissa_bits 尾数位数
 * @param output 输出 fp32 数据
 */
void bfp_to_fp32(const int8_t* mantissas, const int8_t* shared_exps,
                 size_t size, int block_size, int mantissa_bits,
                 float* output);

// ==================== 高级接口 ====================

/**
 * fp32 数组转换为 BFP 并保存（双文件格式，已废弃）
 *
 * @param input fp32 输入数据
 * @param size 数据大小
 * @param mantissa_path mantissa 输出文件路径
 * @param exponent_path shared exponent 输出文件路径
 * @param type BFP 格式类型
 * @return 是否成功
 */
bool save_as_bfp(const float* input, size_t size,
                 const std::string& mantissa_path,
                 const std::string& exponent_path,
                 BFPType type);

/**
 * 从文件加载 BFP 并转换为 fp32（双文件格式，已废弃）
 *
 * @param mantissa_path mantissa 文件路径
 * @param exponent_path shared exponent 文件路径
 * @param type BFP 格式类型
 * @return fp32 数据
 */
std::vector<float> load_bfp_as_fp32(const std::string& mantissa_path,
                                     const std::string& exponent_path,
                                     BFPType type);

/**
 * 从文件加载 BFP 并转换为 fp32（双文件格式，已废弃，指定元素数量）
 */
std::vector<float> load_bfp_as_fp32(const std::string& mantissa_path,
                                     const std::string& exponent_path,
                                     BFPType type,
                                     size_t element_count);

// ==================== 单文件格式接口（推荐） ====================

/**
 * fp32 数组转换为 BFP 并保存（单文件格式）
 *
 * 文件格式: [shared_exps (num_blocks 个 int8)] [mantissas (size 个 int8)]
 *
 * @param input fp32 输入数据
 * @param size 数据大小
 * @param path 输出文件路径
 * @param type BFP 格式类型
 * @return 是否成功
 */
bool save_as_bfp_packed(const float* input, size_t size,
                        const std::string& path, BFPType type);

/**
 * 从单文件加载 BFP 并转换为 fp32
 *
 * 文件格式: [shared_exps (num_blocks 个 int8)] [mantissas (size 个 int8)]
 *
 * @param path 文件路径
 * @param type BFP 格式类型
 * @param element_count 元素数量（用于计算 num_blocks）
 * @return fp32 数据
 */
std::vector<float> load_bfp_packed_as_fp32(const std::string& path,
                                            BFPType type,
                                            size_t element_count);

// ==================== BFP 精度模拟辅助函数 ====================

/**
 * 将单个 fp32 值量化到 BFP 精度（单元素版本）
 * 用于模拟中间计算使用 BFP 精度
 *
 * 注意: BFP 本质上是块级别的量化，单元素量化是近似模拟
 * 这里我们使用自适应指数来模拟 BFP 的精度损失
 */
inline float quantize_fp32_to_bfp_precision(float val, BFPType type) {
    if (std::fabs(val) < 1e-10f) {
        return 0.0f;
    }

    int mantissa_bits = bfp_mantissa_bits(type);
    int8_t max_mantissa = static_cast<int8_t>((1 << (mantissa_bits - 1)) - 1);

    // 计算该值的指数
    int8_t exp = static_cast<int8_t>(std::floor(std::log2(std::fabs(val)))) + 1;

    // 量化
    float scale = std::pow(2.0f, mantissa_bits - 1 - exp);
    int rounded = static_cast<int>(std::round(val * scale));
    rounded = std::max(-static_cast<int>(max_mantissa),
                      std::min(static_cast<int>(max_mantissa), rounded));

    // 反量化
    float inv_scale = std::pow(2.0f, exp - (mantissa_bits - 1));
    return static_cast<float>(rounded) * inv_scale;
}

/**
 * 原地将 fp32 数组量化到 BFP 精度
 */
inline void quantize_inplace_bfp(float* data, size_t size, BFPType type) {
    for (size_t i = 0; i < size; ++i) {
        data[i] = quantize_fp32_to_bfp_precision(data[i], type);
    }
}

/**
 * 将 fp32 数组按块量化到 BFP 精度（真正的块级 BFP 量化）
 * 这是更精确的 BFP 模拟
 */
inline void quantize_block_bfp(float* data, size_t size, BFPType type) {
    int block_size = bfp_block_size(type);
    int mantissa_bits = bfp_mantissa_bits(type);
    int8_t max_mantissa = static_cast<int8_t>((1 << (mantissa_bits - 1)) - 1);

    size_t n_blocks = num_blocks(size, block_size);

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
            shared_exp = -127;
        } else {
            shared_exp = static_cast<int8_t>(std::floor(std::log2(max_abs))) + 1;
        }

        // 3. 量化并反量化每个元素
        float scale = std::pow(2.0f, mantissa_bits - 1 - shared_exp);
        float inv_scale = std::pow(2.0f, shared_exp - (mantissa_bits - 1));

        for (size_t i = start; i < end; ++i) {
            int rounded = static_cast<int>(std::round(data[i] * scale));
            rounded = std::max(-static_cast<int>(max_mantissa),
                              std::min(static_cast<int>(max_mantissa), rounded));
            data[i] = static_cast<float>(rounded) * inv_scale;
        }
    }
}

}  // namespace bfp_io
