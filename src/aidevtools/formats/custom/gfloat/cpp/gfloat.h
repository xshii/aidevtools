/**
 * GFloat Golden API
 *
 * 自定义浮点格式转换的 C++ 实现
 */
#pragma once

#include <cstdint>
#include <vector>

namespace gfloat {

/**
 * fp32 -> gfloat16
 *
 * 格式: 1 符号 + 8 指数 + 7 尾数
 * 取 fp32 高 16 位
 *
 * @param input  输入 fp32 数组
 * @param size   数组大小
 * @param output 输出 uint16 数组 (需预分配)
 */
void fp32_to_gfloat16(const float* input, size_t size, uint16_t* output);

/**
 * gfloat16 -> fp32
 *
 * 低 16 位补零还原
 *
 * @param input  输入 uint16 数组
 * @param size   数组大小
 * @param output 输出 fp32 数组 (需预分配)
 */
void gfloat16_to_fp32(const uint16_t* input, size_t size, float* output);

/**
 * fp32 -> gfloat8
 *
 * 格式: 1 符号 + 4 指数 + 3 尾数
 * 取 fp32 高 8 位
 *
 * @param input  输入 fp32 数组
 * @param size   数组大小
 * @param output 输出 uint8 数组 (需预分配)
 */
void fp32_to_gfloat8(const float* input, size_t size, uint8_t* output);

/**
 * gfloat8 -> fp32
 *
 * 低 24 位补零还原
 *
 * @param input  输入 uint8 数组
 * @param size   数组大小
 * @param output 输出 fp32 数组 (需预分配)
 */
void gfloat8_to_fp32(const uint8_t* input, size_t size, float* output);

// ==================== 向量版本 ====================

std::vector<uint16_t> fp32_to_gfloat16(const std::vector<float>& input);
std::vector<float> gfloat16_to_fp32(const std::vector<uint16_t>& input);
std::vector<uint8_t> fp32_to_gfloat8(const std::vector<float>& input);
std::vector<float> gfloat8_to_fp32(const std::vector<uint8_t>& input);

}  // namespace gfloat
