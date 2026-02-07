#!/usr/bin/env python
"""Optimizer Demo 1: 基础用法

PyTorch 劫持 → Benchmark 提取 → 时延评估

运行: python demos/optimizer/01_basic/run.py
"""
import importlib

mod = importlib.import_module("aidevtools.optimizer.demos.01_basic")

if __name__ == "__main__":
    mod.demo_simple_linear()
    mod.demo_ffn()
    mod.demo_evaluate()
    mod.demo_strategy_compare()
    mod.demo_nn_module()
    print("\n" + "=" * 60)
    print("Demo 完成!")
    print("=" * 60)
