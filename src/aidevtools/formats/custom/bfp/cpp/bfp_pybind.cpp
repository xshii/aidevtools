/**
 * BFP Python bindings using pybind11
 */
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "bfp.h"

namespace py = pybind11;

py::tuple py_fp32_to_bfp(py::array_t<float> input, int block_size, int mantissa_bits) {
    auto buf = input.request();
    float* ptr = static_cast<float*>(buf.ptr);
    size_t size = buf.size;

    size_t n_blocks = bfp::num_blocks(size, block_size);

    // 创建输出数组
    auto mantissas = py::array_t<int8_t>(size);
    auto shared_exps = py::array_t<int8_t>(n_blocks);

    auto m_buf = mantissas.request();
    auto e_buf = shared_exps.request();

    bfp::fp32_to_bfp(ptr, size, block_size, mantissa_bits,
                     static_cast<int8_t*>(m_buf.ptr),
                     static_cast<int8_t*>(e_buf.ptr));

    return py::make_tuple(mantissas, shared_exps);
}

py::array_t<float> py_bfp_to_fp32(py::array_t<int8_t> mantissas,
                                   py::array_t<int8_t> shared_exps,
                                   int block_size, int mantissa_bits) {
    auto m_buf = mantissas.request();
    auto e_buf = shared_exps.request();

    size_t size = m_buf.size;

    auto output = py::array_t<float>(size);
    auto o_buf = output.request();

    bfp::bfp_to_fp32(static_cast<int8_t*>(m_buf.ptr),
                     static_cast<int8_t*>(e_buf.ptr),
                     size, block_size, mantissa_bits,
                     static_cast<float*>(o_buf.ptr));

    return output;
}

PYBIND11_MODULE(bfp_golden, m) {
    m.doc() = "Block Floating Point (BFP) Golden API";

    m.def("fp32_to_bfp", &py_fp32_to_bfp,
          "Convert fp32 to BFP format",
          py::arg("data"),
          py::arg("block_size") = 16,
          py::arg("mantissa_bits") = 8);

    m.def("bfp_to_fp32", &py_bfp_to_fp32,
          "Convert BFP to fp32 format",
          py::arg("mantissas"),
          py::arg("shared_exps"),
          py::arg("block_size") = 16,
          py::arg("mantissa_bits") = 8);
}
