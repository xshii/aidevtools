/**
 * CPU Golden CLI - BFP 版本
 *
 * 通过命令行调用算子，用于 subprocess 方式生成 golden 数据。
 * 使用 BFP (Block Floating Point) 格式。
 *
 * 用法:
 *   ./cpu_golden_bfp <op> <dtype> <input_bin> [weight_bin] <output_bin> <shape...>
 *
 * 示例:
 *   ./cpu_golden_bfp matmul bfp16 a.bin b.bin c.bin 64 128 256
 *   ./cpu_golden_bfp softmax bfp8 input.bin output.bin 4 64
 */
#include <iostream>
#include <string>
#include <vector>
#include <cstdlib>

#include "bfp/io.h"
#include "interface.h"

using namespace bfp_io;
using namespace cpu_golden::ops;

void print_usage(const char* prog) {
    std::cerr << "CPU Golden CLI (BFP)\n\n"
              << "Usage:\n"
              << "  " << prog << " quantize <dtype> <input.bin> <output.bin> <size>\n"
              << "  " << prog << " quantize_block <dtype> <input.bin> <output.bin> <size>\n"
              << "  " << prog << " matmul <dtype> <a.bin> <b.bin> <c.bin> <M> <K> <N>\n"
              << "  " << prog << " softmax <dtype> <input.bin> <output.bin> <batch> <seq>\n"
              << "  " << prog << " layernorm <dtype> <input.bin> <gamma.bin> <beta.bin> <output.bin> <batch> <hidden>\n"
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
              << "Note: quantize converts fp32 to BFP precision (per-element, stored as fp32).\n"
              << "      quantize_block uses true block-level BFP quantization.\n";
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
              << " [" << size << "] (per-element)\n";

    auto input_fp32 = load_fp32(input_path);

    if (input_fp32.size() < size) {
        std::cerr << "Error: input.bin size mismatch\n";
        return 1;
    }

    quantize_inplace_bfp(input_fp32.data(), size, dtype);

    if (!save_fp32(output_path, input_fp32.data(), size)) {
        std::cerr << "Error: failed to save output\n";
        return 1;
    }

    std::cerr << "[cpu_golden_bfp] quantize done: " << output_path << "\n";
    return 0;
}

int run_quantize_block(int argc, char* argv[]) {
    if (argc < 6) {
        std::cerr << "Error: quantize_block requires 4 arguments\n";
        return 1;
    }

    BFPType dtype = parse_bfp_type(argv[2]);
    std::string input_path = argv[3];
    std::string output_path = argv[4];
    size_t size = std::stoull(argv[5]);

    std::cerr << "[cpu_golden_bfp] quantize_block: " << bfp_type_to_string(dtype)
              << " [" << size << "] (block-level)\n";

    auto input_fp32 = load_fp32(input_path);

    if (input_fp32.size() < size) {
        std::cerr << "Error: input.bin size mismatch\n";
        return 1;
    }

    quantize_block_bfp(input_fp32.data(), size, dtype);

    if (!save_fp32(output_path, input_fp32.data(), size)) {
        std::cerr << "Error: failed to save output\n";
        return 1;
    }

    std::cerr << "[cpu_golden_bfp] quantize_block done: " << output_path << "\n";
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

    // BFP 需要 mantissa 和 exponent 文件
    std::string a_exp_path = a_path + ".exp";
    std::string b_exp_path = b_path + ".exp";

    std::cerr << "[cpu_golden] matmul: " << bfp_type_to_string(dtype)
              << " [" << M << "," << K << "] @ [" << K << "," << N << "] (bfp precision)\n";

    auto a_fp32 = load_bfp_as_fp32(a_path, a_exp_path, dtype);
    auto b_fp32 = load_bfp_as_fp32(b_path, b_exp_path, dtype);

    if (a_fp32.size() < M * K) {
        std::cerr << "Error: a.bin size mismatch\n";
        return 1;
    }
    if (b_fp32.size() < K * N) {
        std::cerr << "Error: b.bin size mismatch\n";
        return 1;
    }

    std::vector<float> c_fp32(M * N);
    matmul_bfp(a_fp32.data(), b_fp32.data(), c_fp32.data(), M, K, N, dtype);

    std::string c_exp_path = c_path + ".exp";
    if (!save_as_bfp(c_fp32.data(), M * N, c_path, c_exp_path, dtype)) {
        std::cerr << "Error: failed to save output\n";
        return 1;
    }

    std::cerr << "[cpu_golden] matmul done: " << c_path << "\n";
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

    std::string input_exp_path = input_path + ".exp";

    std::cerr << "[cpu_golden] softmax: " << bfp_type_to_string(dtype)
              << " [" << batch << "," << seq << "] (bfp precision)\n";

    auto input_fp32 = load_bfp_as_fp32(input_path, input_exp_path, dtype);

    if (input_fp32.size() < batch * seq) {
        std::cerr << "Error: input.bin size mismatch\n";
        return 1;
    }

    std::vector<float> output_fp32(batch * seq);
    softmax_bfp(input_fp32.data(), output_fp32.data(), batch, seq, dtype);

    std::string output_exp_path = output_path + ".exp";
    if (!save_as_bfp(output_fp32.data(), batch * seq, output_path, output_exp_path, dtype)) {
        std::cerr << "Error: failed to save output\n";
        return 1;
    }

    std::cerr << "[cpu_golden] softmax done: " << output_path << "\n";
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

    std::cerr << "[cpu_golden] layernorm: " << bfp_type_to_string(dtype)
              << " [" << batch << "," << hidden << "] (bfp precision)\n";

    auto input_fp32 = load_bfp_as_fp32(input_path, input_path + ".exp", dtype);
    auto gamma_fp32 = load_bfp_as_fp32(gamma_path, gamma_path + ".exp", dtype);
    auto beta_fp32 = load_bfp_as_fp32(beta_path, beta_path + ".exp", dtype);

    if (input_fp32.size() < batch * hidden || gamma_fp32.size() < hidden || beta_fp32.size() < hidden) {
        std::cerr << "Error: input size mismatch\n";
        return 1;
    }

    std::vector<float> output_fp32(batch * hidden);
    layernorm_bfp(input_fp32.data(), gamma_fp32.data(), beta_fp32.data(),
                  output_fp32.data(), batch, hidden, 1e-5f, dtype);

    if (!save_as_bfp(output_fp32.data(), batch * hidden, output_path, output_path + ".exp", dtype)) {
        std::cerr << "Error: failed to save output\n";
        return 1;
    }

    std::cerr << "[cpu_golden] layernorm done: " << output_path << "\n";
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

    std::cerr << "[cpu_golden] " << op_name << ": " << bfp_type_to_string(dtype)
              << " [" << size << "] (bfp precision)\n";

    auto input_fp32 = load_bfp_as_fp32(input_path, input_path + ".exp", dtype);

    if (input_fp32.size() < size) {
        std::cerr << "Error: input.bin size mismatch\n";
        return 1;
    }

    std::vector<float> output_fp32(size);
    activation_fn(input_fp32.data(), output_fp32.data(), size, dtype);

    if (!save_as_bfp(output_fp32.data(), size, output_path, output_path + ".exp", dtype)) {
        std::cerr << "Error: failed to save output\n";
        return 1;
    }

    std::cerr << "[cpu_golden] " << op_name << " done: " << output_path << "\n";
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

    std::cerr << "[cpu_golden] " << op_name << ": " << bfp_type_to_string(dtype)
              << " [" << size << "] (bfp precision)\n";

    auto a_fp32 = load_bfp_as_fp32(a_path, a_path + ".exp", dtype);
    auto b_fp32 = load_bfp_as_fp32(b_path, b_path + ".exp", dtype);

    if (a_fp32.size() < size || b_fp32.size() < size) {
        std::cerr << "Error: input size mismatch\n";
        return 1;
    }

    std::vector<float> c_fp32(size);
    elementwise_fn(a_fp32.data(), b_fp32.data(), c_fp32.data(), size, dtype);

    if (!save_as_bfp(c_fp32.data(), size, c_path, c_path + ".exp", dtype)) {
        std::cerr << "Error: failed to save output\n";
        return 1;
    }

    std::cerr << "[cpu_golden] " << op_name << " done: " << c_path << "\n";
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
        } else if (op == "quantize_block") {
            return run_quantize_block(argc, argv);
        } else if (op == "matmul") {
            return run_matmul(argc, argv);
        } else if (op == "softmax") {
            return run_softmax(argc, argv);
        } else if (op == "layernorm") {
            return run_layernorm(argc, argv);
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
