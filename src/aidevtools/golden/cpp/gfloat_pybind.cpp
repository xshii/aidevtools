/**
 * GFloat Python 绑定 (pybind11)
 *
 * 编译方式:
 *   pip install pybind11
 *   c++ -O3 -Wall -shared -std=c++17 -fPIC \
 *       $(python3 -m pybind11 --includes) \
 *       gfloat.cpp gfloat_pybind.cpp \
 *       -o gfloat_golden$(python3-config --extension-suffix)
 */
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "gfloat.h"

namespace py = pybind11;

/**
 * fp32 numpy array -> gfloat16 numpy array
 */
py::array_t<uint16_t> py_fp32_to_gfloat16(py::array_t<float> input) {
    auto buf = input.request();
    float* ptr = static_cast<float*>(buf.ptr);
    size_t size = buf.size;

    // 创建输出数组，保持相同 shape
    py::array_t<uint16_t> output(buf.shape);
    auto out_buf = output.request();
    uint16_t* out_ptr = static_cast<uint16_t*>(out_buf.ptr);

    gfloat::fp32_to_gfloat16(ptr, size, out_ptr);

    return output;
}

/**
 * gfloat16 numpy array -> fp32 numpy array
 */
py::array_t<float> py_gfloat16_to_fp32(py::array_t<uint16_t> input) {
    auto buf = input.request();
    uint16_t* ptr = static_cast<uint16_t*>(buf.ptr);
    size_t size = buf.size;

    py::array_t<float> output(buf.shape);
    auto out_buf = output.request();
    float* out_ptr = static_cast<float*>(out_buf.ptr);

    gfloat::gfloat16_to_fp32(ptr, size, out_ptr);

    return output;
}

/**
 * fp32 numpy array -> gfloat8 numpy array
 */
py::array_t<uint8_t> py_fp32_to_gfloat8(py::array_t<float> input) {
    auto buf = input.request();
    float* ptr = static_cast<float*>(buf.ptr);
    size_t size = buf.size;

    py::array_t<uint8_t> output(buf.shape);
    auto out_buf = output.request();
    uint8_t* out_ptr = static_cast<uint8_t*>(out_buf.ptr);

    gfloat::fp32_to_gfloat8(ptr, size, out_ptr);

    return output;
}

/**
 * gfloat8 numpy array -> fp32 numpy array
 */
py::array_t<float> py_gfloat8_to_fp32(py::array_t<uint8_t> input) {
    auto buf = input.request();
    uint8_t* ptr = static_cast<uint8_t*>(buf.ptr);
    size_t size = buf.size;

    py::array_t<float> output(buf.shape);
    auto out_buf = output.request();
    float* out_ptr = static_cast<float*>(out_buf.ptr);

    gfloat::gfloat8_to_fp32(ptr, size, out_ptr);

    return output;
}

PYBIND11_MODULE(gfloat_golden, m) {
    m.doc() = "GFloat Golden API - C++ implementation";

    m.def("fp32_to_gfloat16", &py_fp32_to_gfloat16,
          py::arg("input"),
          "Convert fp32 numpy array to gfloat16 (uint16)");

    m.def("gfloat16_to_fp32", &py_gfloat16_to_fp32,
          py::arg("input"),
          "Convert gfloat16 (uint16) numpy array to fp32");

    m.def("fp32_to_gfloat8", &py_fp32_to_gfloat8,
          py::arg("input"),
          "Convert fp32 numpy array to gfloat8 (uint8)");

    m.def("gfloat8_to_fp32", &py_gfloat8_to_fp32,
          py::arg("input"),
          "Convert gfloat8 (uint8) numpy array to fp32");
}
