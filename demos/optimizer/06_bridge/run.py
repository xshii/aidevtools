#!/usr/bin/env python
"""Optimizer Demo 6: PyTorch → Benchmark 桥接

从 PyTorch 代码自动提取 Benchmark

运行: python demos/optimizer/06_bridge/run.py
"""
import importlib

mod = importlib.import_module("aidevtools.optimizer.demos.06_bridge")

if __name__ == "__main__":
    mod.demo_concept()
    mod.demo_api()
    mod.demo_workflow()
    mod.demo_comparison()
    print("\n" + "=" * 60)
    print("Demo 完成!")
    print("=" * 60)
