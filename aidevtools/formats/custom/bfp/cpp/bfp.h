/**
 * Block Floating Point (BFP) Golden Implementation
 *
 * BFP 将数据分块，每块共享一个指数，每个元素只存尾数。
 */
#ifndef BFP_H
#define BFP_H

#include <cstdint>
#include <vector>
#include <cmath>
#include <algorithm>

namespace bfp {

/**
 * fp32 -> BFP 量化
 *
 * @param data 输入数据
 * @param size 数据大小
 * @param block_size 块大小
 * @param mantissa_bits 尾数位数
 * @param mantissas 输出尾数数组
 * @param shared_exps 输出共享指数数组
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
 * @param output 输出数据
 */
void bfp_to_fp32(const int8_t* mantissas, const int8_t* shared_exps,
                 size_t size, int block_size, int mantissa_bits,
                 float* output);

/**
 * 计算块数
 */
inline size_t num_blocks(size_t size, int block_size) {
    return (size + block_size - 1) / block_size;
}

} // namespace bfp

#endif // BFP_H
