/**
 * CPU Golden CLI - 命令行接口
 *
 * 通过命令行调用算子，用于 subprocess 方式生成 golden 数据。
 *
 * 架构说明:
 *   - gfloat_io.h/cpp:   GFloat 格式转换与文件 I/O (可替换为 bfp_io 等)
 *   - ops_interface.h:   算子接口定义
 *   - ops_impl.cpp:      算子实现 (可替换为你自己的实现)
 *   - main.cpp:          CLI 框架 (本文件)
 *
 * 用法:
 *   ./cpu_golden <op> <dtype> <input_bin> [weight_bin] <output_bin> <shape...>
 *
 * 示例:
 *   ./cpu_golden matmul gfp16 a.bin b.bin c.bin 64 128 256
 *   ./cpu_golden softmax gfp8 input.bin output.bin 4 64
 *   ./cpu_golden layernorm gfp16 input.bin gamma.bin beta.bin output.bin 4 256
 *   ./cpu_golden transpose gfp16 x.bin y.bin 2 4 8 32
 */
#include <iostream>
#include <string>
#include <vector>
#include <cstdlib>

#include "gfloat/io.h"
#include "interface.h"

using namespace gfloat_io;
using namespace cpu_golden::ops;

void print_usage(const char* prog) {
    std::cerr << "CPU Golden CLI\n\n"
              << "Usage:\n"
              << "  " << prog << " quantize <dtype> <input.bin> <output.bin> <size>\n"
              << "  " << prog << " matmul <dtype> <a.bin> <b.bin> <c.bin> <M> <K> <N>\n"
              << "  " << prog << " matmul_mixed <dtype_a> <dtype_b> <a.bin> <b.bin> <c.bin> <M> <K> <N> <dtype_out>\n"
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
              << "dtype: gfp4, gfp8, gfp16 (or gfloat4, gfloat8, gfloat16, 4, 8, 16)\n"
              << "\n"
              << "Note: quantize converts fp32 to target precision (stored as fp32).\n"
              << "      Use this at data source to avoid repeated conversions.\n"
              << "\n"
              << "Examples:\n"
              << "  " << prog << " quantize gfp16 input.bin quantized.bin 1024\n"
              << "  " << prog << " matmul gfp16 a.bin b.bin c.bin 64 128 256\n"
              << "  " << prog << " softmax gfp8 input.bin output.bin 4 64\n"
              << "  " << prog << " layernorm gfp16 x.bin gamma.bin beta.bin y.bin 4 256\n"
              << "  " << prog << " transpose gfp16 x.bin y.bin 2 4 8 32\n"
              << "  " << prog << " relu gfp16 x.bin y.bin 1024\n"
              << "  " << prog << " gelu gfp16 x.bin y.bin 1024\n"
              << "  " << prog << " add gfp16 a.bin b.bin c.bin 1024\n";
}

// ==================== Quantize 命令 ====================
// 将 fp32 数据量化到目标精度（输出仍是 fp32，但值是量化后的）

int run_quantize(int argc, char* argv[]) {
    if (argc < 6) {
        std::cerr << "Error: quantize requires 4 arguments\n";
        return 1;
    }

    GFloatType dtype = parse_gfloat_type(argv[2]);
    std::string input_path = argv[3];
    std::string output_path = argv[4];
    size_t size = std::stoull(argv[5]);

    std::cerr << "[cpu_golden] quantize: " << gfloat_type_to_string(dtype)
              << " [" << size << "]\n";

    // 直接加载 fp32 文件
    auto input_fp32 = load_fp32(input_path);

    if (input_fp32.size() < size) {
        std::cerr << "Error: input.bin size mismatch, expected " << size
                  << ", got " << input_fp32.size() << "\n";
        return 1;
    }

    // 原地量化到目标精度
    quantize_inplace(input_fp32.data(), size, dtype);

    // 保存为 fp32（值已量化）
    if (!save_fp32(output_path, input_fp32.data(), size)) {
        std::cerr << "Error: failed to save output to " << output_path << "\n";
        return 1;
    }

    std::cerr << "[cpu_golden] quantize done: " << output_path << "\n";
    return 0;
}

int run_matmul(int argc, char* argv[]) {
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

    // 加载输入 (格式转换在 I/O 层完成)
    auto a_fp32 = load_gfloat_as_fp32(a_path, dtype);
    auto b_fp32 = load_gfloat_as_fp32(b_path, dtype);

    if (a_fp32.size() < M * K) {
        std::cerr << "Error: a.bin size mismatch, expected " << M * K << ", got " << a_fp32.size() << "\n";
        return 1;
    }
    if (b_fp32.size() < K * N) {
        std::cerr << "Error: b.bin size mismatch, expected " << K * N << ", got " << b_fp32.size() << "\n";
        return 1;
    }

    // 调用算子 (使用 gfloat 精度模拟硬件行为)
    std::vector<float> c_fp32(M * N);
    matmul_gfloat(a_fp32.data(), b_fp32.data(), c_fp32.data(), M, K, N, dtype);

    // 保存输出 (格式转换在 I/O 层完成)
    if (!save_as_gfloat(c_fp32.data(), M * N, c_path, dtype)) {
        std::cerr << "Error: failed to save output to " << c_path << "\n";
        return 1;
    }

    std::cerr << "[cpu_golden] matmul done: " << c_path << "\n";
    return 0;
}

int run_matmul_mixed(int argc, char* argv[]) {
    if (argc < 11) {
        std::cerr << "Error: matmul_mixed requires 9 arguments\n";
        return 1;
    }

    GFloatType dtype_a = parse_gfloat_type(argv[2]);
    GFloatType dtype_b = parse_gfloat_type(argv[3]);
    std::string a_path = argv[4];
    std::string b_path = argv[5];
    std::string c_path = argv[6];
    size_t M = std::stoull(argv[7]);
    size_t K = std::stoull(argv[8]);
    size_t N = std::stoull(argv[9]);
    GFloatType dtype_out = parse_gfloat_type(argv[10]);

    std::cerr << "[cpu_golden] matmul_mixed: " << gfloat_type_to_string(dtype_a)
              << " x " << gfloat_type_to_string(dtype_b)
              << " -> " << gfloat_type_to_string(dtype_out)
              << " [" << M << "," << K << "] @ [" << K << "," << N << "]\n";

    auto a_fp32 = load_gfloat_as_fp32(a_path, dtype_a);
    auto b_fp32 = load_gfloat_as_fp32(b_path, dtype_b);

    if (a_fp32.size() < M * K) {
        std::cerr << "Error: a.bin size mismatch\n";
        return 1;
    }
    if (b_fp32.size() < K * N) {
        std::cerr << "Error: b.bin size mismatch\n";
        return 1;
    }

    std::vector<float> c_fp32(M * N);
    // 混合精度: 使用输出类型作为计算精度
    matmul_gfloat(a_fp32.data(), b_fp32.data(), c_fp32.data(), M, K, N, dtype_out);

    if (!save_as_gfloat(c_fp32.data(), M * N, c_path, dtype_out)) {
        std::cerr << "Error: failed to save output\n";
        return 1;
    }

    std::cerr << "[cpu_golden] matmul_mixed done: " << c_path << "\n";
    return 0;
}

int run_softmax(int argc, char* argv[]) {
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

    auto input_fp32 = load_gfloat_as_fp32(input_path, dtype);

    if (input_fp32.size() < batch * seq) {
        std::cerr << "Error: input.bin size mismatch\n";
        return 1;
    }

    std::vector<float> output_fp32(batch * seq);
    softmax_gfloat(input_fp32.data(), output_fp32.data(), batch, seq, dtype);

    if (!save_as_gfloat(output_fp32.data(), batch * seq, output_path, dtype)) {
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

    GFloatType dtype = parse_gfloat_type(argv[2]);
    std::string input_path = argv[3];
    std::string gamma_path = argv[4];
    std::string beta_path = argv[5];
    std::string output_path = argv[6];
    size_t batch = std::stoull(argv[7]);
    size_t hidden = std::stoull(argv[8]);

    std::cerr << "[cpu_golden] layernorm: " << gfloat_type_to_string(dtype)
              << " [" << batch << "," << hidden << "]\n";

    auto input_fp32 = load_gfloat_as_fp32(input_path, dtype);
    auto gamma_fp32 = load_gfloat_as_fp32(gamma_path, dtype);
    auto beta_fp32 = load_gfloat_as_fp32(beta_path, dtype);

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

    std::vector<float> output_fp32(batch * hidden);
    layernorm_gfloat(input_fp32.data(), gamma_fp32.data(), beta_fp32.data(),
                     output_fp32.data(), batch, hidden, 1e-5f, dtype);

    if (!save_as_gfloat(output_fp32.data(), batch * hidden, output_path, dtype)) {
        std::cerr << "Error: failed to save output\n";
        return 1;
    }

    std::cerr << "[cpu_golden] layernorm done: " << output_path << "\n";
    return 0;
}

int run_transpose(int argc, char* argv[]) {
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

    auto input_fp32 = load_gfloat_as_fp32(input_path, dtype);

    size_t total = d0 * d1 * d2 * d3;
    if (input_fp32.size() < total) {
        std::cerr << "Error: input.bin size mismatch\n";
        return 1;
    }

    std::vector<float> output_fp32(total);
    transpose_4d_fp32(input_fp32.data(), output_fp32.data(), d0, d1, d2, d3);

    if (!save_as_gfloat(output_fp32.data(), total, output_path, dtype)) {
        std::cerr << "Error: failed to save output\n";
        return 1;
    }

    std::cerr << "[cpu_golden] transpose done: " << output_path << "\n";
    return 0;
}

// ==================== 激活函数通用模板 ====================
// 使用 gfloat 精度版本，模拟硬件行为

template<void (*activation_fn)(const float*, float*, size_t, GFloatType)>
int run_activation(int argc, char* argv[], const char* op_name) {
    if (argc < 6) {
        std::cerr << "Error: " << op_name << " requires 4 arguments\n";
        return 1;
    }

    GFloatType dtype = parse_gfloat_type(argv[2]);
    std::string input_path = argv[3];
    std::string output_path = argv[4];
    size_t size = std::stoull(argv[5]);

    std::cerr << "[cpu_golden] " << op_name << ": " << gfloat_type_to_string(dtype)
              << " [" << size << "] (gfloat precision)\n";

    auto input_fp32 = load_gfloat_as_fp32(input_path, dtype);

    if (input_fp32.size() < size) {
        std::cerr << "Error: input.bin size mismatch\n";
        return 1;
    }

    std::vector<float> output_fp32(size);
    activation_fn(input_fp32.data(), output_fp32.data(), size, dtype);

    if (!save_as_gfloat(output_fp32.data(), size, output_path, dtype)) {
        std::cerr << "Error: failed to save output\n";
        return 1;
    }

    std::cerr << "[cpu_golden] " << op_name << " done: " << output_path << "\n";
    return 0;
}

// ==================== 逐元素运算通用模板 ====================
// 使用 gfloat 精度版本，模拟硬件行为

template<void (*elementwise_fn)(const float*, const float*, float*, size_t, GFloatType)>
int run_elementwise(int argc, char* argv[], const char* op_name) {
    if (argc < 7) {
        std::cerr << "Error: " << op_name << " requires 5 arguments\n";
        return 1;
    }

    GFloatType dtype = parse_gfloat_type(argv[2]);
    std::string a_path = argv[3];
    std::string b_path = argv[4];
    std::string c_path = argv[5];
    size_t size = std::stoull(argv[6]);

    std::cerr << "[cpu_golden] " << op_name << ": " << gfloat_type_to_string(dtype)
              << " [" << size << "] (gfloat precision)\n";

    auto a_fp32 = load_gfloat_as_fp32(a_path, dtype);
    auto b_fp32 = load_gfloat_as_fp32(b_path, dtype);

    if (a_fp32.size() < size || b_fp32.size() < size) {
        std::cerr << "Error: input size mismatch\n";
        return 1;
    }

    std::vector<float> c_fp32(size);
    elementwise_fn(a_fp32.data(), b_fp32.data(), c_fp32.data(), size, dtype);

    if (!save_as_gfloat(c_fp32.data(), size, c_path, dtype)) {
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
        // 激活函数 (使用 gfloat 精度版本)
        } else if (op == "relu") {
            return run_activation<relu_gfloat>(argc, argv, "relu");
        } else if (op == "gelu") {
            return run_activation<gelu_gfloat>(argc, argv, "gelu");
        } else if (op == "sigmoid") {
            return run_activation<sigmoid_gfloat>(argc, argv, "sigmoid");
        } else if (op == "tanh") {
            return run_activation<tanh_gfloat>(argc, argv, "tanh");
        } else if (op == "silu") {
            return run_activation<silu_gfloat>(argc, argv, "silu");
        // 逐元素运算 (使用 gfloat 精度版本)
        } else if (op == "add") {
            return run_elementwise<add_gfloat>(argc, argv, "add");
        } else if (op == "mul") {
            return run_elementwise<mul_gfloat>(argc, argv, "mul");
        } else if (op == "div") {
            return run_elementwise<div_gfloat>(argc, argv, "div");
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
