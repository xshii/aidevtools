/**
 * GFloat CPU Golden Ops
 *
 * 基于 gfloat4/8/16 格式的 CPU 算子实现
 * 用于生成 golden 数据，通过 subprocess 调用
 */
#pragma once

#include <cstdint>
#include <cstddef>
#include <vector>
#include <string>
#include <fstream>
#include <stdexcept>

namespace gfloat_ops {

// ==================== GFloat 格式类型 ====================
enum class GFloatType {
    GFLOAT4,   // 4-bit: 取 fp32 高 4 位
    GFLOAT8,   // 8-bit: 取 fp32 高 8 位
    GFLOAT16   // 16-bit: 取 fp32 高 16 位
};

// 根据字符串获取格式类型
GFloatType parse_gfloat_type(const std::string& type_str);
std::string gfloat_type_to_string(GFloatType type);

// ==================== 模板化文件 I/O ====================

/**
 * 从 binary 文件加载数据 (模板接口)
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
 * 保存数据到 binary 文件 (模板接口)
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

template<typename T>
bool save_binary(const std::string& path, const std::vector<T>& data) {
    return save_binary(path, data.data(), data.size());
}

// 便捷别名
inline std::vector<float> load_fp32(const std::string& path) {
    return load_binary<float>(path);
}

inline std::vector<uint16_t> load_gfloat16(const std::string& path) {
    return load_binary<uint16_t>(path);
}

inline std::vector<uint8_t> load_gfloat8(const std::string& path) {
    return load_binary<uint8_t>(path);
}

inline std::vector<uint8_t> load_gfloat4_packed(const std::string& path) {
    return load_binary<uint8_t>(path);
}

inline bool save_fp32(const std::string& path, const float* data, size_t size) {
    return save_binary(path, data, size);
}

inline bool save_gfloat16(const std::string& path, const uint16_t* data, size_t size) {
    return save_binary(path, data, size);
}

inline bool save_gfloat8(const std::string& path, const uint8_t* data, size_t size) {
    return save_binary(path, data, size);
}

inline bool save_gfloat4_packed(const std::string& path, const uint8_t* data, size_t packed_size) {
    return save_binary(path, data, packed_size);
}

// ==================== GFloat 格式转换 ====================

// gfloat16
void fp32_to_gfloat16(const float* input, size_t size, uint16_t* output);
void gfloat16_to_fp32(const uint16_t* input, size_t size, float* output);

// gfloat8
void fp32_to_gfloat8(const float* input, size_t size, uint8_t* output);
void gfloat8_to_fp32(const uint8_t* input, size_t size, float* output);

// gfloat4 (packed: 2 个 4-bit 值打包到 1 个 uint8_t)
void fp32_to_gfloat4(const float* input, size_t size, uint8_t* output);
void gfloat4_to_fp32(const uint8_t* input, size_t size, float* output);

// 计算打包后的字节数
inline size_t gfloat4_packed_size(size_t element_count) {
    return (element_count + 1) / 2;  // 向上取整
}

// ==================== 通用接口 ====================

/**
 * fp32 数组转换为 gfloat 并保存
 */
bool save_as_gfloat(const float* input, size_t size, const std::string& path, GFloatType type);

/**
 * 从文件加载 gfloat 并转换为 fp32
 */
std::vector<float> load_gfloat_as_fp32(const std::string& path, GFloatType type);

/**
 * 加载 gfloat 文件并指定元素数量 (用于 gfloat4)
 */
std::vector<float> load_gfloat_as_fp32(const std::string& path, GFloatType type, size_t element_count);

// ==================== 算子实现 (fp32 内部计算) ====================

/**
 * MatMul: C = A @ B
 */
void matmul_fp32(const float* a, const float* b, float* c,
                 size_t M, size_t K, size_t N);

/**
 * Softmax: y = softmax(x, axis=-1)
 */
void softmax_fp32(const float* input, float* output,
                  size_t batch, size_t seq);

/**
 * LayerNorm: y = (x - mean) / sqrt(var + eps) * gamma + beta
 */
void layernorm_fp32(const float* input, const float* gamma, const float* beta,
                    float* output, size_t batch, size_t hidden, float eps = 1e-5f);

}  // namespace gfloat_ops
