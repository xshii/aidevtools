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
 * fp32 数组转换为 BFP 并保存
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
 * 从文件加载 BFP 并转换为 fp32
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
 * 从文件加载 BFP 并转换为 fp32 (指定元素数量)
 */
std::vector<float> load_bfp_as_fp32(const std::string& mantissa_path,
                                     const std::string& exponent_path,
                                     BFPType type,
                                     size_t element_count);

}  // namespace bfp_io
