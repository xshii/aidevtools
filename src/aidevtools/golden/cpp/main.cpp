/**
 * GFloat CPU Golden CLI
 *
 * 命令行接口，用于通过 subprocess 调用
 *
 * 用法:
 *   ./cpu_golden <op> <dtype> <input_bin> [weight_bin] <output_bin> <shape...>
 *
 * 示例:
 *   ./cpu_golden matmul gfp16 a.bin b.bin c.bin 64 128 256
 *   ./cpu_golden softmax gfp8 input.bin output.bin 4 64
 *   ./cpu_golden layernorm gfp16 input.bin gamma.bin beta.bin output.bin 4 256
 *
 * 参数说明:
 *   op       - 算子名称: matmul, softmax, layernorm
 *   dtype    - 数据类型: gfp4, gfp8, gfp16 (或 gfloat4, gfloat8, gfloat16, 4, 8, 16)
 *   shape    - 形状参数，具体含义取决于算子:
 *              matmul: M K N
 *              softmax: batch seq
 *              layernorm: batch hidden
 */
#include <iostream>
#include <string>
#include <vector>
#include <cstdlib>
#include "gfloat_ops.h"

using namespace gfloat_ops;

void print_usage(const char* prog) {
    std::cerr << "GFloat CPU Golden CLI\n\n"
              << "Usage:\n"
              << "  " << prog << " matmul <dtype> <a.bin> <b.bin> <c.bin> <M> <K> <N>\n"
              << "  " << prog << " softmax <dtype> <input.bin> <output.bin> <batch> <seq>\n"
              << "  " << prog << " layernorm <dtype> <input.bin> <gamma.bin> <beta.bin> <output.bin> <batch> <hidden>\n"
              << "  " << prog << " transpose <dtype> <input.bin> <output.bin> <d0> <d1> <d2> <d3>\n"
              << "\n"
              << "dtype: gfp4, gfp8, gfp16 (or gfloat4, gfloat8, gfloat16, 4, 8, 16)\n"
              << "\n"
              << "Examples:\n"
              << "  " << prog << " matmul gfp16 a.bin b.bin c.bin 64 128 256\n"
              << "  " << prog << " softmax gfp8 input.bin output.bin 4 64\n"
              << "  " << prog << " layernorm gfp16 x.bin gamma.bin beta.bin y.bin 4 256\n"
              << "  " << prog << " transpose gfp16 x.bin y.bin 2 4 8 32\n";
}

int run_matmul(int argc, char* argv[]) {
    // matmul <dtype> <a.bin> <b.bin> <c.bin> <M> <K> <N>
    if (argc < 9) {
        std::cerr << "Error: matmul requires 7 arguments\n";
        return 1;
    }

    GFloatType dtype = parse_gfloat_type(argv[2]);
    std::string a_path = argv[3];
    std::string b_path = argv[4];
    std::string c_path = argv[5];
    size_t M = std::stoull(argv[6]);
    size_t K = std::stoull(argv[7]);
    size_t N = std::stoull(argv[8]);

    std::cerr << "[cpu_golden] matmul: " << gfloat_type_to_string(dtype)
              << " [" << M << "," << K << "] @ [" << K << "," << N << "]\n";

    // 加载输入
    auto a_fp32 = load_gfloat_as_fp32(a_path, dtype);
    auto b_fp32 = load_gfloat_as_fp32(b_path, dtype);

    // 检查大小
    if (a_fp32.size() < M * K) {
        std::cerr << "Error: a.bin size mismatch, expected " << M * K << ", got " << a_fp32.size() << "\n";
        return 1;
    }
    if (b_fp32.size() < K * N) {
        std::cerr << "Error: b.bin size mismatch, expected " << K * N << ", got " << b_fp32.size() << "\n";
        return 1;
    }

    // 计算
    std::vector<float> c_fp32(M * N);
    matmul_fp32(a_fp32.data(), b_fp32.data(), c_fp32.data(), M, K, N);

    // 保存输出
    if (!save_as_gfloat(c_fp32.data(), M * N, c_path, dtype)) {
        std::cerr << "Error: failed to save output to " << c_path << "\n";
        return 1;
    }

    std::cerr << "[cpu_golden] matmul done: " << c_path << "\n";
    return 0;
}

int run_softmax(int argc, char* argv[]) {
    // softmax <dtype> <input.bin> <output.bin> <batch> <seq>
    if (argc < 7) {
        std::cerr << "Error: softmax requires 5 arguments\n";
        return 1;
    }

    GFloatType dtype = parse_gfloat_type(argv[2]);
    std::string input_path = argv[3];
    std::string output_path = argv[4];
    size_t batch = std::stoull(argv[5]);
    size_t seq = std::stoull(argv[6]);

    std::cerr << "[cpu_golden] softmax: " << gfloat_type_to_string(dtype)
              << " [" << batch << "," << seq << "]\n";

    // 加载输入
    auto input_fp32 = load_gfloat_as_fp32(input_path, dtype);

    // 检查大小
    if (input_fp32.size() < batch * seq) {
        std::cerr << "Error: input.bin size mismatch, expected " << batch * seq
                  << ", got " << input_fp32.size() << "\n";
        return 1;
    }

    // 计算
    std::vector<float> output_fp32(batch * seq);
    softmax_fp32(input_fp32.data(), output_fp32.data(), batch, seq);

    // 保存输出
    if (!save_as_gfloat(output_fp32.data(), batch * seq, output_path, dtype)) {
        std::cerr << "Error: failed to save output to " << output_path << "\n";
        return 1;
    }

    std::cerr << "[cpu_golden] softmax done: " << output_path << "\n";
    return 0;
}

int run_layernorm(int argc, char* argv[]) {
    // layernorm <dtype> <input.bin> <gamma.bin> <beta.bin> <output.bin> <batch> <hidden>
    if (argc < 9) {
        std::cerr << "Error: layernorm requires 7 arguments\n";
        return 1;
    }

    GFloatType dtype = parse_gfloat_type(argv[2]);
    std::string input_path = argv[3];
    std::string gamma_path = argv[4];
    std::string beta_path = argv[5];
    std::string output_path = argv[6];
    size_t batch = std::stoull(argv[7]);
    size_t hidden = std::stoull(argv[8]);

    std::cerr << "[cpu_golden] layernorm: " << gfloat_type_to_string(dtype)
              << " [" << batch << "," << hidden << "]\n";

    // 加载输入
    auto input_fp32 = load_gfloat_as_fp32(input_path, dtype);
    auto gamma_fp32 = load_gfloat_as_fp32(gamma_path, dtype);
    auto beta_fp32 = load_gfloat_as_fp32(beta_path, dtype);

    // 检查大小
    if (input_fp32.size() < batch * hidden) {
        std::cerr << "Error: input.bin size mismatch\n";
        return 1;
    }
    if (gamma_fp32.size() < hidden) {
        std::cerr << "Error: gamma.bin size mismatch\n";
        return 1;
    }
    if (beta_fp32.size() < hidden) {
        std::cerr << "Error: beta.bin size mismatch\n";
        return 1;
    }

    // 计算
    std::vector<float> output_fp32(batch * hidden);
    layernorm_fp32(input_fp32.data(), gamma_fp32.data(), beta_fp32.data(),
                   output_fp32.data(), batch, hidden);

    // 保存输出
    if (!save_as_gfloat(output_fp32.data(), batch * hidden, output_path, dtype)) {
        std::cerr << "Error: failed to save output to " << output_path << "\n";
        return 1;
    }

    std::cerr << "[cpu_golden] layernorm done: " << output_path << "\n";
    return 0;
}

int run_transpose(int argc, char* argv[]) {
    // transpose <dtype> <input.bin> <output.bin> <d0> <d1> <d2> <d3>
    if (argc < 9) {
        std::cerr << "Error: transpose requires 7 arguments\n";
        return 1;
    }

    GFloatType dtype = parse_gfloat_type(argv[2]);
    std::string input_path = argv[3];
    std::string output_path = argv[4];
    size_t d0 = std::stoull(argv[5]);
    size_t d1 = std::stoull(argv[6]);
    size_t d2 = std::stoull(argv[7]);
    size_t d3 = std::stoull(argv[8]);

    std::cerr << "[cpu_golden] transpose: " << gfloat_type_to_string(dtype)
              << " [" << d0 << "," << d1 << "," << d2 << "," << d3 << "] -> ["
              << d0 << "," << d1 << "," << d3 << "," << d2 << "]\n";

    // 加载输入
    auto input_fp32 = load_gfloat_as_fp32(input_path, dtype);

    size_t total = d0 * d1 * d2 * d3;
    if (input_fp32.size() < total) {
        std::cerr << "Error: input.bin size mismatch, expected " << total
                  << ", got " << input_fp32.size() << "\n";
        return 1;
    }

    // 计算
    std::vector<float> output_fp32(total);
    transpose_4d_fp32(input_fp32.data(), output_fp32.data(), d0, d1, d2, d3);

    // 保存输出
    if (!save_as_gfloat(output_fp32.data(), total, output_path, dtype)) {
        std::cerr << "Error: failed to save output to " << output_path << "\n";
        return 1;
    }

    std::cerr << "[cpu_golden] transpose done: " << output_path << "\n";
    return 0;
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        print_usage(argv[0]);
        return 1;
    }

    std::string op = argv[1];

    try {
        if (op == "matmul") {
            return run_matmul(argc, argv);
        } else if (op == "softmax") {
            return run_softmax(argc, argv);
        } else if (op == "layernorm") {
            return run_layernorm(argc, argv);
        } else if (op == "transpose") {
            return run_transpose(argc, argv);
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
