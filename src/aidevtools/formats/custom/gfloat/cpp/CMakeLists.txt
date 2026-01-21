cmake_minimum_required(VERSION 3.15)
project(gfloat_golden)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# 查找 Python
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

# 查找 pybind11
execute_process(
    COMMAND ${Python3_EXECUTABLE} -m pybind11 --cmakedir
    OUTPUT_VARIABLE pybind11_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
find_package(pybind11 CONFIG REQUIRED)

# 创建 Python 模块
pybind11_add_module(gfloat_golden
    gfloat.cpp
    gfloat_pybind.cpp
)

# 设置输出目录
set_target_properties(gfloat_golden PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/.."
)

# 安装
install(TARGETS gfloat_golden DESTINATION .)
