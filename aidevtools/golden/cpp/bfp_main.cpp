/**
 * CPU Golden CLI - BFP 版本
 *
 * 通过命令行调用算子，用于 subprocess 方式生成 golden 数据。
 * 使用 BFP (Block Floating Point) 格式。
 *
 * I/O 格式（单文件格式）:
 *   - 文件格式: [shared_exps (num_blocks 个 int8)] [mantissas (size 个 int8)]
 *   - 输入输出均使用此单文件格式
 *
 * 用法:
 *   ./cpu_golden_bfp <op> <dtype> <input_bin> [weight_bin] <output_bin> <shape...>
 *
 * 示例:
 *   ./cpu_golden_bfp matmul bfp16 a.bin b.bin c.bin 64 128 256
 *   ./cpu_golden_bfp softmax bfp8 input.bin output.bin 4 64
 *   ./cpu_golden_bfp transpose bfp16 x.bin y.bin 2 4 8 32
 */
#include <iostream>
#include <string>
#include <vector>
#include <cstdlib>

#include "bfp/io.h"
#include "interface.h"

using namespace bfp_io;
using namespace cpu_golden::ops;

// 获取较高精度 (bfp16 > bfp8 > bfp4)
BFPType max_precision(BFPType a, BFPType b) {
    // BFP16 > BFP8 > BFP4 (按 mantissa_bits: 8 > 4 > 2)
    int bits_a = bfp_mantissa_bits(a);
    int bits_b = bfp_mantissa_bits(b);
    return (bits_a >= bits_b) ? a : b;
}

void print_usage(const char* prog) {
    std::cerr << "CPU Golden CLI (BFP)\n\n"
              << "Usage:\n"
              << "  " << prog << " quantize <dtype> <input.bin> <output.bin> <size>\n"
              << "  " << prog << " encode <dtype> <input_fp32.bin> <output_packed.bin> <size>\n"
              << "  " << prog << " decode <dtype> <input_packed.bin> <output_fp32.bin> <size>\n"
              << "  " << prog << " matmul <dtype> <a.bin> <b.bin> <c.bin> <M> <K> <N>\n"
              << "  " << prog << " matmul_mixed <dtype_a> <dtype_b> <a.bin> <b.bin> <c.bin> <M> <K> <N>\n"
              << "  " << prog << " softmax <dtype> <input.bin> <output.bin> <batch> <seq>\n"
              << "  " << prog << " layernorm <dtype> <input.bin> <gamma.bin> <beta.bin> <output.bin> <batch> <hidden>\n"
              << "  " << prog << " transpose <dtype> <input.bin> <output.bin> <d0> <d1> <d2> <d3>\n"
              << "  " << prog << " relu <dtype> <input.bin> <output.bin> <size>\n"
              << "  " << prog << " gelu <dtype> <input.bin> <output.bin> <size>\n"
              << "  " << prog << " sigmoid <dtype> <input.bin> <output.bin> <size>\n"
              << "  " << prog << " tanh <dtype> <input.bin> <output.bin> <size>\n"
              << "  " << prog << " silu <dtype> <input.bin> <output.bin> <size>\n"
              << "  " << prog << " add <dtype> <a.bin> <b.bin> <c.bin> <size>\n"
              << "  " << prog << " mul <dtype> <a.bin> <b.bin> <c.bin> <size>\n"
              << "  " << prog << " div <dtype> <a.bin> <b.bin> <c.bin> <size>\n"
              << "\n"
              << "dtype: bfp4, bfp8, bfp16\n"
              << "\n"
              << "matmul_mixed: compute precision = max(dtype_a, dtype_b), output uses compute precision\n"
              << "File format: [shared_exps (num_blocks)] [mantissas (size)] in single file\n";
}

// ==================== Quantize 命令 ====================

int run_quantize(int argc, char* argv[]) {
    if (argc < 6) {
        std::cerr << "Error: quantize requires 4 arguments\n";
        return 1;
    }

    BFPType dtype = parse_bfp_type(argv[2]);
    std::string input_path = argv[3];
    std::string output_path = argv[4];
    size_t size = std::stoull(argv[5]);

    std::cerr << "[cpu_golden_bfp] quantize: " << bfp_type_to_string(dtype)
              << " [" << size << "]\n";

    // 直接加载 fp32 文件（与 GFloat 版本一致）
    auto input_fp32 = load_fp32(input_path);

    if (input_fp32.size() < size) {
        std::cerr << "Error: input.bin size mismatch, expected " << size
                  << ", got " << input_fp32.size() << "\n";
        return 1;
    }

    // 原地量化到 BFP 精度（按块量化）
    quantize_block_bfp(input_fp32.data(), size, dtype);

    // 保存为 fp32（值已量化）
    if (!save_fp32(output_path, input_fp32.data(), size)) {
        std::cerr << "Error: failed to save output\n";
        return 1;
    }

    std::cerr << "[cpu_golden_bfp] quantize done: " << output_path << "\n";
    return 0;
}

// ==================== Encode 命令 (fp32 -> BFP packed) ====================

int run_encode(int argc, char* argv[]) {
    if (argc < 6) {
        std::cerr << "Error: encode requires 4 arguments\n";
        return 1;
    }

    BFPType dtype = parse_bfp_type(argv[2]);
    std::string input_path = argv[3];
    std::string output_path = argv[4];
    size_t size = std::stoull(argv[5]);

    std::cerr << "[cpu_golden_bfp] encode: " << bfp_type_to_string(dtype)
              << " [" << size << "] fp32 -> packed\n";

    // 读取 fp32
    auto input_fp32 = load_fp32(input_path);
    if (input_fp32.size() < size) {
        std::cerr << "Error: input size mismatch\n";
        return 1;
    }

    // 保存为 BFP packed 格式
    if (!save_as_bfp_packed(input_fp32.data(), size, output_path, dtype)) {
        std::cerr << "Error: failed to save output\n";
        return 1;
    }

    std::cerr << "[cpu_golden_bfp] encode done: " << output_path << "\n";
    return 0;
}

// ==================== Decode 命令 (BFP packed -> fp32) ====================

int run_decode(int argc, char* argv[]) {
    if (argc < 6) {
        std::cerr << "Error: decode requires 4 arguments\n";
        return 1;
    }

    BFPType dtype = parse_bfp_type(argv[2]);
    std::string input_path = argv[3];
    std::string output_path = argv[4];
    size_t size = std::stoull(argv[5]);

    std::cerr << "[cpu_golden_bfp] decode: " << bfp_type_to_string(dtype)
              << " [" << size << "] packed -> fp32\n";

    // 读取 BFP packed
    auto output_fp32 = load_bfp_packed_as_fp32(input_path, dtype, size);

    // 保存为 fp32
    if (!save_fp32(output_path, output_fp32.data(), size)) {
        std::cerr << "Error: failed to save output\n";
        return 1;
    }

    std::cerr << "[cpu_golden_bfp] decode done: " << output_path << "\n";
    return 0;
}

int run_matmul(int argc, char* argv[]) {
    if (argc < 9) {
        std::cerr << "Error: matmul requires 7 arguments\n";
        return 1;
    }

    BFPType dtype = parse_bfp_type(argv[2]);
    std::string a_path = argv[3];
    std::string b_path = argv[4];
    std::string c_path = argv[5];
    size_t M = std::stoull(argv[6]);
    size_t K = std::stoull(argv[7]);
    size_t N = std::stoull(argv[8]);

    std::cerr << "[cpu_golden_bfp] matmul: " << bfp_type_to_string(dtype)
              << " [" << M << "," << K << "] @ [" << K << "," << N << "]\n";

    // 读取单文件 BFP 格式输入
    auto a_fp32 = load_bfp_packed_as_fp32(a_path, dtype, M * K);
    auto b_fp32 = load_bfp_packed_as_fp32(b_path, dtype, K * N);

    std::vector<float> c_fp32(M * N);
    matmul_bfp(a_fp32.data(), b_fp32.data(), c_fp32.data(), M, K, N, dtype);

    // 保存单文件 BFP 格式输出
    if (!save_as_bfp_packed(c_fp32.data(), M * N, c_path, dtype)) {
        std::cerr << "Error: failed to save output\n";
        return 1;
    }

    std::cerr << "[cpu_golden_bfp] matmul done: " << c_path << "\n";
    return 0;
}

int run_matmul_mixed(int argc, char* argv[]) {
    // 接受 9 或 10 个参数（dtype_out 可选，会被忽略）
    if (argc < 10) {
        std::cerr << "Error: matmul_mixed requires at least 8 arguments\n";
        return 1;
    }

    BFPType dtype_a = parse_bfp_type(argv[2]);
    BFPType dtype_b = parse_bfp_type(argv[3]);
    std::string a_path = argv[4];
    std::string b_path = argv[5];
    std::string c_path = argv[6];
    size_t M = std::stoull(argv[7]);
    size_t K = std::stoull(argv[8]);
    size_t N = std::stoull(argv[9]);

    // 计算精度 = max(dtype_a, dtype_b)，忽略用户指定的 dtype_out
    BFPType dtype_compute = max_precision(dtype_a, dtype_b);

    // 如果用户指定了 dtype_out，检查是否与计算精度一致
    if (argc >= 11) {
        BFPType dtype_out = parse_bfp_type(argv[10]);
        if (dtype_out != dtype_compute) {
            std::cerr << "[cpu_golden_bfp] warning: dtype_out=" << bfp_type_to_string(dtype_out)
                      << " ignored, using compute precision " << bfp_type_to_string(dtype_compute) << "\n";
        }
    }

    std::cerr << "[cpu_golden_bfp] matmul_mixed: " << bfp_type_to_string(dtype_a)
              << " x " << bfp_type_to_string(dtype_b)
              << " -> " << bfp_type_to_string(dtype_compute)
              << " [" << M << "," << K << "] @ [" << K << "," << N << "]\n";

    // 各自用自己的精度读取
    auto a_fp32 = load_bfp_packed_as_fp32(a_path, dtype_a, M * K);
    auto b_fp32 = load_bfp_packed_as_fp32(b_path, dtype_b, K * N);

    std::vector<float> c_fp32(M * N);
    // 用较高精度计算
    matmul_bfp(a_fp32.data(), b_fp32.data(), c_fp32.data(), M, K, N, dtype_compute);

    // 用计算精度保存
    if (!save_as_bfp_packed(c_fp32.data(), M * N, c_path, dtype_compute)) {
        std::cerr << "Error: failed to save output\n";
        return 1;
    }

    std::cerr << "[cpu_golden_bfp] matmul_mixed done: " << c_path << "\n";
    return 0;
}

int run_softmax(int argc, char* argv[]) {
    if (argc < 7) {
        std::cerr << "Error: softmax requires 5 arguments\n";
        return 1;
    }

    BFPType dtype = parse_bfp_type(argv[2]);
    std::string input_path = argv[3];
    std::string output_path = argv[4];
    size_t batch = std::stoull(argv[5]);
    size_t seq = std::stoull(argv[6]);

    std::cerr << "[cpu_golden_bfp] softmax: " << bfp_type_to_string(dtype)
              << " [" << batch << "," << seq << "]\n";

    auto input_fp32 = load_bfp_packed_as_fp32(input_path, dtype, batch * seq);

    std::vector<float> output_fp32(batch * seq);
    softmax_bfp(input_fp32.data(), output_fp32.data(), batch, seq, dtype);

    if (!save_as_bfp_packed(output_fp32.data(), batch * seq, output_path, dtype)) {
        std::cerr << "Error: failed to save output\n";
        return 1;
    }

    std::cerr << "[cpu_golden_bfp] softmax done: " << output_path << "\n";
    return 0;
}

int run_layernorm(int argc, char* argv[]) {
    if (argc < 9) {
        std::cerr << "Error: layernorm requires 7 arguments\n";
        return 1;
    }

    BFPType dtype = parse_bfp_type(argv[2]);
    std::string input_path = argv[3];
    std::string gamma_path = argv[4];
    std::string beta_path = argv[5];
    std::string output_path = argv[6];
    size_t batch = std::stoull(argv[7]);
    size_t hidden = std::stoull(argv[8]);

    std::cerr << "[cpu_golden_bfp] layernorm: " << bfp_type_to_string(dtype)
              << " [" << batch << "," << hidden << "]\n";

    auto input_fp32 = load_bfp_packed_as_fp32(input_path, dtype, batch * hidden);
    auto gamma_fp32 = load_bfp_packed_as_fp32(gamma_path, dtype, hidden);
    auto beta_fp32 = load_bfp_packed_as_fp32(beta_path, dtype, hidden);

    std::vector<float> output_fp32(batch * hidden);
    layernorm_bfp(input_fp32.data(), gamma_fp32.data(), beta_fp32.data(),
                  output_fp32.data(), batch, hidden, 1e-5f, dtype);

    if (!save_as_bfp_packed(output_fp32.data(), batch * hidden, output_path, dtype)) {
        std::cerr << "Error: failed to save output\n";
        return 1;
    }

    std::cerr << "[cpu_golden_bfp] layernorm done: " << output_path << "\n";
    return 0;
}

int run_transpose(int argc, char* argv[]) {
    if (argc < 9) {
        std::cerr << "Error: transpose requires 7 arguments\n";
        return 1;
    }

    BFPType dtype = parse_bfp_type(argv[2]);
    std::string input_path = argv[3];
    std::string output_path = argv[4];
    size_t d0 = std::stoull(argv[5]);
    size_t d1 = std::stoull(argv[6]);
    size_t d2 = std::stoull(argv[7]);
    size_t d3 = std::stoull(argv[8]);
    size_t total_size = d0 * d1 * d2 * d3;

    std::cerr << "[cpu_golden_bfp] transpose: " << bfp_type_to_string(dtype)
              << " [" << d0 << "," << d1 << "," << d2 << "," << d3 << "] -> "
              << "[" << d0 << "," << d1 << "," << d3 << "," << d2 << "]\n";

    auto input_fp32 = load_bfp_packed_as_fp32(input_path, dtype, total_size);

    std::vector<float> output_fp32(total_size);
    transpose_4d_fp32(input_fp32.data(), output_fp32.data(), d0, d1, d2, d3);

    if (!save_as_bfp_packed(output_fp32.data(), total_size, output_path, dtype)) {
        std::cerr << "Error: failed to save output\n";
        return 1;
    }

    std::cerr << "[cpu_golden_bfp] transpose done: " << output_path << "\n";
    return 0;
}

// 激活函数通用模板
template<void (*activation_fn)(const float*, float*, size_t, BFPType)>
int run_activation(int argc, char* argv[], const char* op_name) {
    if (argc < 6) {
        std::cerr << "Error: " << op_name << " requires 4 arguments\n";
        return 1;
    }

    BFPType dtype = parse_bfp_type(argv[2]);
    std::string input_path = argv[3];
    std::string output_path = argv[4];
    size_t size = std::stoull(argv[5]);

    std::cerr << "[cpu_golden_bfp] " << op_name << ": " << bfp_type_to_string(dtype)
              << " [" << size << "]\n";

    auto input_fp32 = load_bfp_packed_as_fp32(input_path, dtype, size);

    std::vector<float> output_fp32(size);
    activation_fn(input_fp32.data(), output_fp32.data(), size, dtype);

    if (!save_as_bfp_packed(output_fp32.data(), size, output_path, dtype)) {
        std::cerr << "Error: failed to save output\n";
        return 1;
    }

    std::cerr << "[cpu_golden_bfp] " << op_name << " done: " << output_path << "\n";
    return 0;
}

// 逐元素运算通用模板
template<void (*elementwise_fn)(const float*, const float*, float*, size_t, BFPType)>
int run_elementwise(int argc, char* argv[], const char* op_name) {
    if (argc < 7) {
        std::cerr << "Error: " << op_name << " requires 5 arguments\n";
        return 1;
    }

    BFPType dtype = parse_bfp_type(argv[2]);
    std::string a_path = argv[3];
    std::string b_path = argv[4];
    std::string c_path = argv[5];
    size_t size = std::stoull(argv[6]);

    std::cerr << "[cpu_golden_bfp] " << op_name << ": " << bfp_type_to_string(dtype)
              << " [" << size << "]\n";

    auto a_fp32 = load_bfp_packed_as_fp32(a_path, dtype, size);
    auto b_fp32 = load_bfp_packed_as_fp32(b_path, dtype, size);

    std::vector<float> c_fp32(size);
    elementwise_fn(a_fp32.data(), b_fp32.data(), c_fp32.data(), size, dtype);

    if (!save_as_bfp_packed(c_fp32.data(), size, c_path, dtype)) {
        std::cerr << "Error: failed to save output\n";
        return 1;
    }

    std::cerr << "[cpu_golden_bfp] " << op_name << " done: " << c_path << "\n";
    return 0;
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        print_usage(argv[0]);
        return 1;
    }

    std::string op = argv[1];

    try {
        if (op == "quantize") {
            return run_quantize(argc, argv);
        } else if (op == "encode") {
            return run_encode(argc, argv);
        } else if (op == "decode") {
            return run_decode(argc, argv);
        } else if (op == "matmul") {
            return run_matmul(argc, argv);
        } else if (op == "matmul_mixed") {
            return run_matmul_mixed(argc, argv);
        } else if (op == "softmax") {
            return run_softmax(argc, argv);
        } else if (op == "layernorm") {
            return run_layernorm(argc, argv);
        } else if (op == "transpose") {
            return run_transpose(argc, argv);
        // 激活函数
        } else if (op == "relu") {
            return run_activation<relu_bfp>(argc, argv, "relu");
        } else if (op == "gelu") {
            return run_activation<gelu_bfp>(argc, argv, "gelu");
        } else if (op == "sigmoid") {
            return run_activation<sigmoid_bfp>(argc, argv, "sigmoid");
        } else if (op == "tanh") {
            return run_activation<tanh_bfp>(argc, argv, "tanh");
        } else if (op == "silu") {
            return run_activation<silu_bfp>(argc, argv, "silu");
        // 逐元素运算
        } else if (op == "add") {
            return run_elementwise<add_bfp>(argc, argv, "add");
        } else if (op == "mul") {
            return run_elementwise<mul_bfp>(argc, argv, "mul");
        } else if (op == "div") {
            return run_elementwise<div_bfp>(argc, argv, "div");
        } else if (op == "-h" || op == "--help" || op == "help") {
            print_usage(argv[0]);
            return 0;
        } else {
            std::cerr << "Error: unknown op '" << op << "'\n";
            print_usage(argv[0]);
            return 1;
        }
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
}
