#!/usr/bin/env python
"""Optimizer Demo 4: ECharts 可视化

图表生成: 柱状图、饼图、折线图、Roofline、热力图

运行: python demos/optimizer/04_echarts/run.py
"""
import importlib

mod = importlib.import_module("aidevtools.optimizer.demos.04_echarts")

if __name__ == "__main__":
    mod.demo_bar_chart()
    mod.demo_pie_chart()
    mod.demo_line_chart()
    mod.demo_roofline()
    mod.demo_gauge()
    mod.demo_heatmap()
    mod.demo_from_evaluator()
    mod.demo_save_html()
    print("\n" + "=" * 60)
    print("Demo 完成!")
    print("=" * 60)
