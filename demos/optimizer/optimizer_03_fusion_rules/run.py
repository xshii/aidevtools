#!/usr/bin/env python
"""Optimizer Demo 3: 融合规则配置

全局规则、多算子模式、超参数配置

运行: python demos/optimizer/03_fusion_rules/run.py
"""
import importlib

mod = importlib.import_module("aidevtools.optimizer.demos.03_fusion_rules")

if __name__ == "__main__":
    mod.demo_global_rules()
    mod.demo_add_custom_rule()
    mod.demo_fusion_patterns()
    mod.demo_auto_composition()
    mod.demo_hyper_params()
    mod.demo_override_pair()
    print("\n" + "=" * 60)
    print("Demo 完成!")
    print("=" * 60)
