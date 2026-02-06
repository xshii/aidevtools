"""
PyTorch 劫持 → Benchmark 桥接

从 ops 模块的计算图自动生成 Benchmark，替代手动链式构建。

使用示例:
    import aidevtools.golden as golden
    import torch.nn.functional as F

    # 执行 PyTorch 代码 (被劫持)
    x = torch.randn(512, 768)
    w1 = torch.randn(768, 3072)
    w2 = torch.randn(3072, 768)

    y = F.linear(x, w1.T)
    y = F.gelu(y)
    y = F.linear(y, w2.T)

    # 自动提取为 Benchmark
    from aidevtools.optimizer.bridge import extract_benchmark

    bm = extract_benchmark(name="auto_ffn")
    print(bm.summary())

    # 直接进行时延评估
    from aidevtools.optimizer import FusionEvaluator
    evaluator = FusionEvaluator()
    result = evaluator.evaluate(bm)
"""

from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass

from aidevtools.ops import get_graph, get_graph_ops, OpNode
from .benchmark import Benchmark, OpSpec, OpType


# OpNode.op_type → OpType 映射
OP_TYPE_MAP = {
    "matmul": OpType.MATMUL,
    "linear": OpType.MATMUL,
    "mm": OpType.MATMUL,
    "bmm": OpType.MATMUL,
    "conv2d": OpType.CONV,
    "conv1d": OpType.CONV,
    "gelu": OpType.GELU,
    "relu": OpType.RELU,
    "silu": OpType.SILU,
    "softmax": OpType.SOFTMAX,
    "layer_norm": OpType.LAYERNORM,
    "layernorm": OpType.LAYERNORM,
    "rms_norm": OpType.RMSNORM,
    "rmsnorm": OpType.RMSNORM,
    "add": OpType.ADD,
    "mul": OpType.MUL,
    "transpose": OpType.TRANSPOSE,
    "reshape": OpType.RESHAPE,
    "embedding": OpType.EMBEDDING,
    "attention": OpType.ATTENTION,
}


def _infer_shapes(node: OpNode) -> Dict[str, int]:
    """从 OpNode 推断 shapes"""
    shapes = {}

    # 从 input_data 推断
    for key, data in node.input_data.items():
        if hasattr(data, 'shape'):
            shape = data.shape
            if len(shape) >= 2:
                if node.op_type in ("matmul", "linear", "mm"):
                    # matmul: (M, K) @ (K, N) -> (M, N)
                    if key in ("input", "x", "a"):
                        shapes["M"] = shape[-2]
                        shapes["K"] = shape[-1]
                    elif key in ("weight", "w", "b"):
                        if len(shape) == 2:
                            shapes["K"] = shape[0]
                            shapes["N"] = shape[1]
                        else:
                            shapes["N"] = shape[-1]
                elif node.op_type in ("gelu", "relu", "silu"):
                    shapes["M"] = shape[-2] if len(shape) >= 2 else 1
                    shapes["N"] = shape[-1]
                elif node.op_type == "softmax":
                    shapes["M"] = shape[-2] if len(shape) >= 2 else 1
                    shapes["N"] = shape[-1]

    # 从 output_data 推断
    if node.output_data is not None:
        out_shape = node.output_data.shape
        if len(out_shape) >= 2:
            if "M" not in shapes:
                shapes["M"] = out_shape[-2]
            if "N" not in shapes:
                shapes["N"] = out_shape[-1]

    # 默认值
    shapes.setdefault("M", 1)
    shapes.setdefault("N", 1)
    shapes.setdefault("K", shapes.get("N", 1))

    return shapes


def _node_to_opspec(node: OpNode) -> Optional[OpSpec]:
    """将 OpNode 转换为 OpSpec"""
    op_type = OP_TYPE_MAP.get(node.op_type.lower())
    if op_type is None:
        # 未知算子类型，跳过
        return None

    shapes = _infer_shapes(node)

    return OpSpec(
        name=node.name,
        op_type=op_type,
        shapes=shapes,
    )


def extract_benchmark(name: str = "auto",
                      clear_graph: bool = False) -> Benchmark:
    """
    从 ops 计算图或记录自动提取 Benchmark

    支持两种模式：
    1. FULL_GRAPH/MIXED 模式: 从计算图提取（包含完整依赖关系）
    2. SINGLE_OP 模式: 从记录提取（根据执行顺序）

    Args:
        name: Benchmark 名称
        clear_graph: 提取后是否清空

    Returns:
        Benchmark 对象

    示例:
        import aidevtools.golden
        import torch.nn.functional as F

        y = F.linear(x, w1)
        y = F.gelu(y)
        y = F.linear(y, w2)

        bm = extract_benchmark("my_ffn")
        print(bm.summary())
    """
    from aidevtools.ops import clear as ops_clear, get_records

    bm = Benchmark(name)

    # 优先从计算图提取（FULL_GRAPH/MIXED 模式）
    graph = get_graph()
    ops_list = get_graph_ops()

    if ops_list:
        for op_name in ops_list:
            node = graph.get(op_name)
            if node is None:
                continue

            op_spec = _node_to_opspec(node)
            if op_spec is None:
                continue

            bm.ops.append(op_spec)
    else:
        # 从记录提取（SINGLE_OP 模式）
        records = get_records()
        for record in records:
            op_spec = _record_to_opspec(record)
            if op_spec is None:
                continue
            bm.ops.append(op_spec)

    if clear_graph:
        ops_clear()

    return bm


def _record_to_opspec(record: Dict[str, Any]) -> Optional[OpSpec]:
    """将 record 转换为 OpSpec"""
    op_name = record.get("op", "")
    full_name = record.get("name", "")

    op_type = OP_TYPE_MAP.get(op_name.lower())
    if op_type is None:
        return None

    # 推断 shapes
    shapes = {}
    input_data = record.get("input")
    weight_data = record.get("weight")
    golden_data = record.get("golden")

    if input_data is not None and hasattr(input_data, 'shape'):
        shape = input_data.shape
        if len(shape) >= 2:
            shapes["M"] = shape[-2]
            shapes["K"] = shape[-1]

    if weight_data is not None and hasattr(weight_data, 'shape'):
        shape = weight_data.shape
        if len(shape) >= 2:
            shapes["K"] = shape[0]
            shapes["N"] = shape[1]

    if golden_data is not None and hasattr(golden_data, 'shape'):
        shape = golden_data.shape
        if len(shape) >= 2:
            if "M" not in shapes:
                shapes["M"] = shape[-2]
            if "N" not in shapes:
                shapes["N"] = shape[-1]

    shapes.setdefault("M", 1)
    shapes.setdefault("N", 1)
    shapes.setdefault("K", shapes.get("N", 1))

    return OpSpec(
        name=full_name,
        op_type=op_type,
        shapes=shapes,
    )


def extract_and_evaluate(name: str = "auto",
                         strategies: Optional[List[str]] = None,
                         clear_graph: bool = True):
    """
    一键: 提取 Benchmark + 评估

    Args:
        name: Benchmark 名称
        strategies: 策略列表 (默认全部)
        clear_graph: 提取后是否清空计算图

    Returns:
        EvalResult 或 CompareResult

    示例:
        import aidevtools.golden
        import torch.nn.functional as F

        y = F.linear(x, w1)
        y = F.gelu(y)
        y = F.linear(y, w2)

        result = extract_and_evaluate("my_ffn")
        print(result.summary())
    """
    from .evaluator import FusionEvaluator

    bm = extract_benchmark(name, clear_graph)
    evaluator = FusionEvaluator()

    if strategies:
        return evaluator.compare(bm, strategies=strategies)
    else:
        return evaluator.evaluate(bm)


@dataclass
class TracedBenchmark:
    """带追踪信息的 Benchmark"""
    benchmark: Benchmark
    graph: Dict[str, OpNode]
    op_order: List[str]

    def summary(self) -> str:
        lines = [
            f"TracedBenchmark: {self.benchmark.name}",
            f"  算子数: {len(self.benchmark.ops)}",
            f"  追踪节点: {len(self.graph)}",
            "",
            "算子列表:",
        ]
        for op in self.benchmark.ops:
            lines.append(f"  - {op.name}: {op.op_type.value} {op.shapes}")
        return "\n".join(lines)


def trace_and_extract(func, *args, name: str = "traced", **kwargs) -> TracedBenchmark:
    """
    执行函数并提取 Benchmark

    Args:
        func: 要执行的函数
        *args: 函数参数
        name: Benchmark 名称
        **kwargs: 函数关键字参数

    Returns:
        TracedBenchmark

    示例:
        def my_ffn(x, w1, w2):
            y = F.linear(x, w1)
            y = F.gelu(y)
            y = F.linear(y, w2)
            return y

        result = trace_and_extract(my_ffn, x, w1, w2, name="ffn")
        print(result.summary())
    """
    from aidevtools.ops import clear as ops_clear

    # 清空之前的图
    ops_clear()

    # 执行函数
    func(*args, **kwargs)

    # 提取
    graph = get_graph().copy()
    op_order = get_graph_ops().copy()
    bm = extract_benchmark(name, clear_graph=False)

    return TracedBenchmark(
        benchmark=bm,
        graph=graph,
        op_order=op_order,
    )
