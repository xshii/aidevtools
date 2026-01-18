"""算子注册表

定义 xlsx 模板中可用的算子列表及其元信息。
"""
from typing import Dict, List, Any

# 默认算子注册表
# key: 算子名
# value: 算子元信息
_default_ops: Dict[str, Dict[str, Any]] = {
    "linear": {
        "inputs": ["x", "weight"],
        "optional": ["bias"],
        "description": "线性变换 y = x @ weight + bias",
    },
    "matmul": {
        "inputs": ["a", "b"],
        "optional": [],
        "description": "矩阵乘法 c = a @ b",
    },
    "conv2d": {
        "inputs": ["x", "weight"],
        "optional": ["bias", "stride", "padding"],
        "description": "2D 卷积",
    },
    "relu": {
        "inputs": ["x"],
        "optional": [],
        "description": "ReLU 激活",
    },
    "softmax": {
        "inputs": ["x"],
        "optional": ["axis"],
        "description": "Softmax 激活",
    },
    "layernorm": {
        "inputs": ["x", "weight", "bias"],
        "optional": ["eps"],
        "description": "Layer Normalization",
    },
    "batchnorm": {
        "inputs": ["x", "mean", "var", "weight", "bias"],
        "optional": ["eps"],
        "description": "Batch Normalization",
    },
    "attention": {
        "inputs": ["q", "k", "v"],
        "optional": ["mask", "scale"],
        "description": "Scaled Dot-Product Attention",
    },
    "add": {
        "inputs": ["a", "b"],
        "optional": [],
        "description": "逐元素加法",
    },
    "mul": {
        "inputs": ["a", "b"],
        "optional": [],
        "description": "逐元素乘法",
    },
    "transpose": {
        "inputs": ["x"],
        "optional": ["axes"],
        "description": "转置",
    },
    "reshape": {
        "inputs": ["x"],
        "optional": ["shape"],
        "description": "形状变换",
    },
    "concat": {
        "inputs": ["tensors"],
        "optional": ["axis"],
        "description": "张量拼接",
    },
    "split": {
        "inputs": ["x"],
        "optional": ["num_splits", "axis"],
        "description": "张量分割",
    },
    "pooling": {
        "inputs": ["x"],
        "optional": ["kernel_size", "stride", "mode"],
        "description": "池化 (max/avg)",
    },
}


def get_default_ops() -> Dict[str, Dict[str, Any]]:
    """获取默认算子注册表"""
    return _default_ops.copy()


def get_op_info(name: str) -> Dict[str, Any]:
    """获取算子信息"""
    if name not in _default_ops:
        return {"inputs": ["x"], "optional": [], "description": f"自定义算子: {name}"}
    return _default_ops[name]


def list_ops() -> List[str]:
    """列出所有注册的算子"""
    return list(_default_ops.keys())


def validate_op(name: str) -> bool:
    """检查算子是否有效"""
    return name in _default_ops
