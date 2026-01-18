cmake_minimum_required(VERSION 3.14)
project(bfp_golden)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# 查找 pybind11
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
execute_process(
    COMMAND ${Python3_EXECUTABLE} -m pybind11 --cmakedir
    OUTPUT_VARIABLE pybind11_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
find_package(pybind11 CONFIG REQUIRED)

# 编译模块
pybind11_add_module(bfp_golden
    bfp.cpp
    bfp_pybind.cpp
)

# 输出到 golden 目录
set_target_properties(bfp_golden PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/..
)
