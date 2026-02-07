#!/usr/bin/env python
"""Optimizer Demo 2: ML 校准流程

使用 PyTorch 生成测试用例、导入实测数据并校准超参数

运行: python demos/optimizer/02_calibration/run.py
"""
import importlib

mod = importlib.import_module("aidevtools.optimizer.demos.02_calibration")

if __name__ == "__main__":
    test_cases = mod.demo_generate_test_cases()
    mod.demo_export_for_dut(test_cases)
    archive = mod.demo_import_results()
    mod.demo_calibration(archive)
    mod.demo_workflow()
    print("\n" + "=" * 60)
    print("Demo 完成!")
    print("=" * 60)
