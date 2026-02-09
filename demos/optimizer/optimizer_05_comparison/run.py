#!/usr/bin/env python
"""Optimizer Demo 5: 理论 vs 工程化对比

理论分析 vs ML 校准后的工程化方法精度对比

运行: python demos/optimizer/05_comparison/run.py
"""
import importlib

mod = importlib.import_module("aidevtools.optimizer.demos.05_comparison")

if __name__ == "__main__":
    mod.demo_simple_compare()
    mod.demo_detailed_analysis()
    mod.demo_scenario_analysis()
    mod.demo_generate_echarts()
    mod.demo_calibrate_and_compare()
    print("\n" + "=" * 70)
    print("Demo 完成!")
    print("=" * 70)
