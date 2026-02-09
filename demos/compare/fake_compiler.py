#!/usr/bin/env python
"""假编译脚本 - 模拟将 Model DSL 编译到 cpu_golden 后端

设计：
  - 输入：Model DSL 定义（Python 代码）
  - 输出：可执行的编译产物（实际调用 cpu_golden）

流程：
  1. 解析 Model DSL（实际上是 Python 对象）
  2. 生成 cpu_golden 调用序列
  3. 返回可执行函数
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import numpy as np
from typing import Dict, Any, Callable


class FakeCompiler:
    """假编译器 - 将 Model DSL 编译为 cpu_golden 可执行"""

    def __init__(self):
        self.cpu_golden_available = self._check_cpu_golden()

    def _check_cpu_golden(self) -> bool:
        """检查 cpu_golden 是否可用"""
        try:
            # 尝试导入 cpu_golden
            from aidevtools.backend import cpu_golden
            return True
        except ImportError:
            return False

    def compile(self, model_fn: Callable) -> Callable:
        """编译 Model 函数到 cpu_golden 后端

        Args:
            model_fn: 返回 Model 对象的函数

        Returns:
            compiled_fn: 接受 inputs 字典，返回 outputs 字典
        """
        if not self.cpu_golden_available:
            # 如果 cpu_golden 不可用，fallback 到 PyTorch
            print("  [假编译] cpu_golden 不可用，使用 PyTorch fallback")
            return self._compile_to_pytorch(model_fn)

        print("  [假编译] 编译到 cpu_golden 后端")
        return self._compile_to_cpu_golden(model_fn)

    def _compile_to_cpu_golden(self, model_fn: Callable) -> Callable:
        """编译到 cpu_golden 后端"""
        from aidevtools.backend import cpu_golden

        def compiled_fn(inputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
            """执行编译后的模型（使用 cpu_golden）"""
            model = model_fn()
            outputs = {}

            # 获取所有操作
            for op_name in model.operations.keys():
                op_def = model.operations[op_name]
                op_type = op_def["type"]

                # 获取输入
                input_names = op_def.get("inputs", [])
                op_inputs = []
                for inp_name in input_names:
                    if inp_name in inputs:
                        op_inputs.append(inputs[inp_name])
                    elif inp_name in outputs:
                        op_inputs.append(outputs[inp_name])
                    else:
                        raise ValueError(f"找不到输入: {inp_name}")

                # 调用 cpu_golden 算子
                if op_type == "matmul":
                    result = cpu_golden.matmul(op_inputs[0], op_inputs[1])
                elif op_type == "linear":
                    weight = op_def.get("weight")
                    bias = op_def.get("bias")
                    result = cpu_golden.linear(op_inputs[0], weight, bias)
                elif op_type == "gelu":
                    result = cpu_golden.gelu(op_inputs[0])
                elif op_type == "relu":
                    result = cpu_golden.relu(op_inputs[0])
                elif op_type == "silu":
                    result = cpu_golden.silu(op_inputs[0])
                elif op_type == "softmax":
                    dim = op_def.get("dim", -1)
                    result = cpu_golden.softmax(op_inputs[0], dim=dim)
                elif op_type == "layernorm":
                    eps = op_def.get("eps", 1e-5)
                    result = cpu_golden.layernorm(op_inputs[0], eps=eps)
                elif op_type == "add":
                    result = cpu_golden.add(op_inputs[0], op_inputs[1])
                elif op_type == "mul":
                    result = cpu_golden.mul(op_inputs[0], op_inputs[1])
                else:
                    raise ValueError(f"不支持的算子类型: {op_type}")

                outputs[op_name] = result

            return outputs

        return compiled_fn

    def _compile_to_pytorch(self, model_fn: Callable) -> Callable:
        """Fallback: 编译到 PyTorch（如果 cpu_golden 不可用）"""
        import torch
        import torch.nn.functional as F

        def compiled_fn(inputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
            """执行编译后的模型（使用 PyTorch）"""
            model = model_fn()
            outputs = {}

            # 转换输入为 torch tensor
            torch_inputs = {k: torch.from_numpy(v) for k, v in inputs.items()}

            # 获取所有操作
            for op_name in model.operations.keys():
                op_def = model.operations[op_name]
                op_type = op_def["type"]

                # 获取输入
                input_names = op_def.get("inputs", [])
                op_inputs = []
                for inp_name in input_names:
                    if inp_name in torch_inputs:
                        op_inputs.append(torch_inputs[inp_name])
                    elif inp_name in outputs:
                        op_inputs.append(outputs[inp_name])
                    else:
                        raise ValueError(f"找不到输入: {inp_name}")

                # 调用 PyTorch 算子
                if op_type == "matmul":
                    result = torch.matmul(op_inputs[0], op_inputs[1])
                elif op_type == "linear":
                    weight = torch.from_numpy(op_def.get("weight"))
                    bias = op_def.get("bias")
                    if bias is not None:
                        bias = torch.from_numpy(bias)
                    result = F.linear(op_inputs[0], weight, bias)
                elif op_type == "gelu":
                    result = F.gelu(op_inputs[0])
                elif op_type == "relu":
                    result = F.relu(op_inputs[0])
                elif op_type == "silu":
                    result = F.silu(op_inputs[0])
                elif op_type == "softmax":
                    dim = op_def.get("dim", -1)
                    result = F.softmax(op_inputs[0], dim=dim)
                elif op_type == "layernorm":
                    normalized_shape = op_inputs[0].shape[-1:]
                    eps = op_def.get("eps", 1e-5)
                    result = F.layer_norm(op_inputs[0], normalized_shape, eps=eps)
                elif op_type == "add":
                    result = torch.add(op_inputs[0], op_inputs[1])
                elif op_type == "mul":
                    result = torch.mul(op_inputs[0], op_inputs[1])
                else:
                    raise ValueError(f"不支持的算子类型: {op_type}")

                outputs[op_name] = result

            # 转换输出为 numpy
            np_outputs = {k: v.detach().numpy().astype(np.float32) for k, v in outputs.items()}
            return np_outputs

        return compiled_fn


def compile_model(model_fn: Callable) -> Callable:
    """快捷函数：编译 Model 到 cpu_golden 后端"""
    compiler = FakeCompiler()
    return compiler.compile(model_fn)
