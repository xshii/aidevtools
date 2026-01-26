"""PyTorch 后端集成

将 aidevtools 的能力集成到 PyTorch 工作流中：
- 性能分析 (Paper Analysis)
- 精确比对 / 模糊比对 / 量化比对
- 自定义数据类型 (bfp/gfp)
- 可选流程开关
- 前向/反向传播 golden 支持

用法:
    from aidevtools.torch_backend import golden_mode

    # 推理模式
    with golden_mode(golden="python", compare="fuzzy", quantize="gfp16") as backend:
        y = model(x)
        backend.print_comparison_report()

    # 训练模式 (前向+反向都走 golden)
    with golden_mode(golden="python", backward=True) as backend:
        y = model(x)
        loss = y.sum()
        loss.backward()  # 反向传播也走 golden

    # 性能分析
    with golden_mode(golden="python", profile=True) as backend:
        y = model(x)
        profiles = backend.get_profiles()
"""

import warnings
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Literal, Optional

import numpy as np
import torch
import torch.nn.functional as F_torch

from aidevtools.analysis.profile import OpProfile
from aidevtools.formats.quantize import simulate_quantize
from aidevtools.ops import _functional as F_golden
from aidevtools.ops.base import set_golden_mode
from aidevtools.ops.cpu_golden import set_cpu_golden_dtype
from aidevtools.tools.compare.diff import compare_full


@dataclass
class CompareResult:
    """比对结果"""
    op_name: str
    exact_match: bool
    max_abs_error: float
    mean_abs_error: float
    cosine_sim: float
    qsnr_db: float
    status: str  # "PASS" | "FAIL" | "QUANT_ISSUE"
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TorchBackendConfig:
    """后端配置"""
    # Golden 模式
    golden_mode: Literal["cpp", "python", "none"] = "python"

    # 比对模式
    compare_mode: Literal["exact", "fuzzy", "quantized", "none"] = "none"
    compare_rtol: float = 1e-5
    compare_atol: float = 1e-5

    # 量化配置
    quantize_type: Optional[str] = None  # "gfp16" | "gfp8" | "bfp16" | "bfp8" | None
    quantize_input: bool = False  # 是否量化输入
    quantize_output: bool = False  # 是否量化输出

    # Profile 配置
    profile_enabled: bool = False

    # 反向传播配置
    backward_enabled: bool = False  # 是否劫持反向传播

    # 调试
    verbose: bool = False


# ============================================================
# 自定义 Autograd Functions (支持前向/反向 golden)
# ============================================================


class GoldenLinearFunction(torch.autograd.Function):
    """Linear 前向/反向 golden"""

    @staticmethod
    def forward(ctx, input, weight, bias=None):
        # 保存用于反向传播
        ctx.save_for_backward(input, weight, bias)

        # 前向: y = x @ W.T + b
        input_np = input.detach().cpu().numpy()
        weight_np = weight.detach().cpu().numpy()
        bias_np = bias.detach().cpu().numpy() if bias is not None else None

        output = F_golden.linear(input_np, weight_np, bias_np)
        return torch.from_numpy(output).to(input.device)

    @staticmethod
    def backward(ctx, grad_output):
        input, weight, bias = ctx.saved_tensors
        grad_output_np = grad_output.detach().cpu().numpy()
        input_np = input.detach().cpu().numpy()
        weight_np = weight.detach().cpu().numpy()

        # 反向传播公式:
        # grad_input = grad_output @ weight
        # grad_weight = grad_output.T @ input
        # grad_bias = grad_output.sum(dim=0)

        grad_input = None
        grad_weight = None
        grad_bias = None

        if ctx.needs_input_grad[0]:
            # grad_input = grad_output @ weight
            grad_input_np = F_golden.matmul(grad_output_np, weight_np)
            grad_input = torch.from_numpy(grad_input_np).to(input.device)

        if ctx.needs_input_grad[1]:
            # grad_weight = grad_output.T @ input (reshape for batch)
            # 对于 [B, *, out] @ [B, *, in] -> [out, in]
            go_flat = grad_output_np.reshape(-1, grad_output_np.shape[-1])
            in_flat = input_np.reshape(-1, input_np.shape[-1])
            grad_weight_np = F_golden.matmul(go_flat.T, in_flat)
            grad_weight = torch.from_numpy(grad_weight_np).to(weight.device)

        if bias is not None and ctx.needs_input_grad[2]:
            # grad_bias = sum over batch dims
            grad_bias_np = grad_output_np.reshape(-1, grad_output_np.shape[-1]).sum(axis=0)
            grad_bias = torch.from_numpy(grad_bias_np.astype(np.float32)).to(bias.device)

        return grad_input, grad_weight, grad_bias


class GoldenReLUFunction(torch.autograd.Function):
    """ReLU 前向/反向 golden"""

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        input_np = input.detach().cpu().numpy()
        output = F_golden.relu(input_np)
        return torch.from_numpy(output).to(input.device)

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_output_np = grad_output.detach().cpu().numpy()
        input_np = input.detach().cpu().numpy()

        # ReLU 反向: grad_input = grad_output * (input > 0)
        grad_input_np = grad_output_np * (input_np > 0).astype(np.float32)
        return torch.from_numpy(grad_input_np).to(input.device)


class GoldenGELUFunction(torch.autograd.Function):
    """GELU 前向/反向 golden"""

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        input_np = input.detach().cpu().numpy()
        output = F_golden.gelu(input_np)
        return torch.from_numpy(output).to(input.device)

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        # GELU 的精确梯度较复杂，使用 torch 计算
        input.requires_grad_(True)
        with torch.enable_grad():
            y = F_torch.gelu(input)
            grad_input, = torch.autograd.grad(y, input, grad_output)
        return grad_input


class GoldenSoftmaxFunction(torch.autograd.Function):
    """Softmax 前向/反向 golden"""

    @staticmethod
    def forward(ctx, input, dim=-1):
        input_np = input.detach().cpu().numpy()
        output = F_golden.softmax(input_np, dim=dim)
        output_tensor = torch.from_numpy(output).to(input.device)
        ctx.save_for_backward(output_tensor)
        ctx.dim = dim
        return output_tensor

    @staticmethod
    def backward(ctx, grad_output):
        output, = ctx.saved_tensors
        dim = ctx.dim

        # Softmax 反向: grad_input = output * (grad_output - sum(grad_output * output))
        grad_output_np = grad_output.detach().cpu().numpy()
        output_np = output.detach().cpu().numpy()

        sum_term = np.sum(grad_output_np * output_np, axis=dim, keepdims=True)
        grad_input_np = output_np * (grad_output_np - sum_term)

        return torch.from_numpy(grad_input_np.astype(np.float32)).to(grad_output.device), None


# Autograd function 注册表
AUTOGRAD_FUNCTIONS = {
    'linear': GoldenLinearFunction,
    'relu': GoldenReLUFunction,
    'gelu': GoldenGELUFunction,
    'softmax': GoldenSoftmaxFunction,
}


class TorchGoldenBackend:
    """PyTorch Golden 后端

    劫持 torch.nn.functional 调用，支持：
    1. 替换为 CPU golden (cpp/python)
    2. 精确/模糊/量化比对
    3. 性能分析 (Paper Analysis)
    4. 自定义数据类型
    """

    # 支持的算子映射: torch_fn -> (golden_fn, op_type, flops_fn)
    OP_REGISTRY = {
        'linear': {
            'torch_fn': F_torch.linear,
            'golden_fn': F_golden.linear,
            'op_type': 'linear',
            'compute_unit': 'cube',
            'flops': lambda s: 2 * s['M'] * s['K'] * s['N'],
        },
        'relu': {
            'torch_fn': F_torch.relu,
            'golden_fn': F_golden.relu,
            'op_type': 'relu',
            'compute_unit': 'vector',
            'flops': lambda s: s['input_size'],
        },
        'gelu': {
            'torch_fn': F_torch.gelu,
            'golden_fn': F_golden.gelu,
            'op_type': 'gelu',
            'compute_unit': 'vector',
            'flops': lambda s: 8 * s['input_size'],
        },
        'softmax': {
            'torch_fn': F_torch.softmax,
            'golden_fn': F_golden.softmax,
            'op_type': 'softmax',
            'compute_unit': 'vector',
            'flops': lambda s: 5 * s['input_size'],
        },
        'layer_norm': {
            'torch_fn': F_torch.layer_norm,
            'golden_fn': F_golden.layer_norm,
            'op_type': 'layernorm',
            'compute_unit': 'vector',
            'flops': lambda s: 8 * s['input_size'],
        },
        'silu': {
            'torch_fn': F_torch.silu,
            'golden_fn': F_golden.silu,
            'op_type': 'silu',
            'compute_unit': 'vector',
            'flops': lambda s: 4 * s['input_size'],
        },
        'sigmoid': {
            'torch_fn': F_torch.sigmoid,
            'golden_fn': F_golden.sigmoid,
            'op_type': 'sigmoid',
            'compute_unit': 'vector',
            'flops': lambda s: 4 * s['input_size'],
        },
        'tanh': {
            'torch_fn': F_torch.tanh,
            'golden_fn': F_golden.tanh,
            'op_type': 'tanh',
            'compute_unit': 'vector',
            'flops': lambda s: 4 * s['input_size'],
        },
    }

    def __init__(self, config: TorchBackendConfig = None):
        self.config = config or TorchBackendConfig()
        self._originals: Dict[str, Callable] = {}
        self._enabled = False
        self._op_counter: Dict[str, int] = {}
        self._in_golden_call = False  # 防止递归

        # 结果收集
        self._profiles: List[OpProfile] = []
        self._compare_results: List[CompareResult] = []
        self._records: List[Dict[str, Any]] = []

    def configure(self, **kwargs):
        """更新配置"""
        for k, v in kwargs.items():
            if hasattr(self.config, k):
                setattr(self.config, k, v)
            else:
                raise ValueError(f"未知配置项: {k}")

        # 同步到全局配置
        if 'golden_mode' in kwargs:
            set_golden_mode(kwargs['golden_mode'])
        if 'quantize_type' in kwargs and kwargs['quantize_type']:
            # 设置 CPU golden dtype
            qtype = kwargs['quantize_type']
            if qtype in ('gfp16', 'gfp8', 'gfp4'):
                set_cpu_golden_dtype(qtype.replace('gfp', 'gfp'))

    def enable(self):
        """启用后端"""
        if self._enabled:
            return

        for name, op_info in self.OP_REGISTRY.items():
            self._originals[name] = op_info['torch_fn']
            wrapper = self._make_wrapper(name, op_info)
            setattr(F_torch, name, wrapper)

        self._enabled = True
        if self.config.verbose:
            print(f"[TorchGoldenBackend] 已启用 (golden={self.config.golden_mode}, "
                  f"compare={self.config.compare_mode}, quantize={self.config.quantize_type})")

    def disable(self):
        """禁用后端"""
        if not self._enabled:
            return

        for name, original in self._originals.items():
            setattr(F_torch, name, original)

        self._originals.clear()
        self._enabled = False
        if self.config.verbose:
            print("[TorchGoldenBackend] 已禁用")

    def clear(self):
        """清空记录"""
        self._profiles.clear()
        self._compare_results.clear()
        self._records.clear()
        self._op_counter.clear()

    def _make_wrapper(self, op_name: str, op_info: dict):
        """创建劫持 wrapper"""
        golden_fn = op_info['golden_fn']
        original_fn = op_info['torch_fn']

        def wrapper(*args, **kwargs):
            # 防止递归: 如果正在执行 golden，直接调用原始函数
            if self._in_golden_call:
                return original_fn(*args, **kwargs)

            # 计数
            idx = self._op_counter.get(op_name, 0)
            self._op_counter[op_name] = idx + 1
            full_name = f"{op_name}_{idx}"

            # 如果启用反向传播且有对应的 autograd function
            if self.config.backward_enabled and op_name in AUTOGRAD_FUNCTIONS:
                autograd_fn = AUTOGRAD_FUNCTIONS[op_name]
                result = autograd_fn.apply(*args, **kwargs)

                # Profile (Paper Analysis)
                if self.config.profile_enabled:
                    np_args, _ = self._to_numpy_args(args, kwargs)
                    output_np = result.detach().cpu().numpy()
                    profile = self._create_profile(full_name, op_info, np_args, kwargs, output_np)
                    self._profiles.append(profile)

                # 记录
                self._records.append({
                    'name': full_name,
                    'op_type': op_name,
                    'input_shapes': [a.shape for a in args if hasattr(a, 'shape')],
                    'output_shape': result.shape,
                    'backward_enabled': True,
                })

                return result

            # 非反向传播模式: 只劫持前向
            # 转换为 numpy
            np_args, np_kwargs = self._to_numpy_args(args, kwargs)

            # 可选: 量化输入
            if self.config.quantize_input and self.config.quantize_type:
                np_args = tuple(
                    simulate_quantize(a, self.config.quantize_type)
                    if isinstance(a, np.ndarray) else a
                    for a in np_args
                )

            # 计算 golden (设置标志防止递归)
            if self.config.golden_mode != "none":
                try:
                    self._in_golden_call = True
                    golden_result = golden_fn(*np_args, **np_kwargs)
                except Exception as e:
                    warnings.warn(f"[{full_name}] Golden 计算失败: {e}")
                    golden_result = None
                finally:
                    self._in_golden_call = False
            else:
                golden_result = None

            # 计算 reference (原始 torch)
            with torch.no_grad():
                ref_result = original_fn(*args, **kwargs)
            ref_np = ref_result.detach().cpu().numpy()

            # 可选: 量化输出
            if self.config.quantize_output and self.config.quantize_type:
                if golden_result is not None:
                    golden_result = simulate_quantize(golden_result, self.config.quantize_type)

            # 比对
            if self.config.compare_mode != "none" and golden_result is not None:
                compare_result = self._compare(full_name, golden_result, ref_np)
                self._compare_results.append(compare_result)

            # Profile (Paper Analysis)
            if self.config.profile_enabled:
                profile = self._create_profile(full_name, op_info, np_args, np_kwargs, golden_result)
                self._profiles.append(profile)

            # 记录
            self._records.append({
                'name': full_name,
                'op_type': op_name,
                'input_shapes': [a.shape for a in np_args if isinstance(a, np.ndarray)],
                'output_shape': golden_result.shape if golden_result is not None else ref_np.shape,
            })

            # 返回结果
            if golden_result is not None:
                return torch.from_numpy(golden_result.astype(np.float32))
            return ref_result

        return wrapper

    def _to_numpy_args(self, args, kwargs):
        """Tensor -> numpy"""
        np_args = tuple(
            a.detach().cpu().numpy().astype(np.float32) if isinstance(a, torch.Tensor) else a
            for a in args
        )
        np_kwargs = {
            k: v.detach().cpu().numpy().astype(np.float32) if isinstance(v, torch.Tensor) else v
            for k, v in kwargs.items()
        }
        return np_args, np_kwargs

    def _compare(self, name: str, golden: np.ndarray, reference: np.ndarray) -> CompareResult:
        """比对 golden 和 reference"""
        # 使用现有的比对工具
        result = compare_full(golden, reference, atol=self.config.compare_atol, rtol=self.config.compare_rtol)

        # 判断状态
        exact_match = result.passed and result.max_abs < 1e-6
        if self.config.compare_mode == "exact":
            status = "PASS" if exact_match else "FAIL"
        elif self.config.compare_mode == "fuzzy":
            status = "PASS" if result.cosine > 0.999 else "FAIL"
        elif self.config.compare_mode == "quantized":
            # 量化比对: 允许更大误差
            status = "PASS" if result.qsnr > 20 else "FAIL"
            if status != "PASS" and result.qsnr > 10:
                status = "QUANT_ISSUE"
        else:
            status = "UNKNOWN"

        return CompareResult(
            op_name=name,
            exact_match=exact_match,
            max_abs_error=result.max_abs,
            mean_abs_error=result.mean_abs,
            cosine_sim=result.cosine,
            qsnr_db=result.qsnr,
            status=status,
        )

    def _create_profile(self, name: str, op_info: dict, args, kwargs, output) -> OpProfile:
        """创建 OpProfile"""
        # 计算 shapes
        input_shapes = [a.shape for a in args if isinstance(a, np.ndarray)]
        output_shape = output.shape if output is not None else None

        # 计算 sizes
        input_size = sum(np.prod(s) for s in input_shapes)
        output_size = np.prod(output_shape) if output_shape else 0

        # 计算 FLOPs
        shapes_dict = {
            'input_size': input_size,
            'output_size': output_size,
        }
        if op_info['op_type'] == 'linear' and len(input_shapes) >= 2:
            # linear: input [*, K], weight [N, K]
            shapes_dict['M'] = int(np.prod(input_shapes[0][:-1]))
            shapes_dict['K'] = input_shapes[0][-1]
            shapes_dict['N'] = input_shapes[1][0]

        flops = op_info['flops'](shapes_dict) if 'flops' in op_info else 0

        return OpProfile(
            name=name,
            op_type=op_info['op_type'],
            compute_unit=op_info.get('compute_unit', 'vector'),
            flops=flops,
            input_bytes=int(input_size * 4),  # fp32
            output_bytes=int(output_size * 4),
            weight_bytes=0,
        )

    # === 结果获取 ===

    def get_profiles(self) -> List[OpProfile]:
        """获取 Paper Analysis profiles"""
        return self._profiles.copy()

    def get_compare_results(self) -> List[CompareResult]:
        """获取比对结果"""
        return self._compare_results.copy()

    def get_records(self) -> List[Dict]:
        """获取记录"""
        return self._records.copy()

    def print_comparison_report(self):
        """打印比对报告"""
        if not self._compare_results:
            print("无比对结果")
            return

        print("=" * 80)
        print(f"{'op_name':<20} {'exact':^6} {'max_err':>10} {'cosine':>10} {'qsnr':>8} {'status':^10}")
        print("-" * 80)

        pass_count = sum(1 for r in self._compare_results if r.status == "PASS")
        fail_count = sum(1 for r in self._compare_results if r.status == "FAIL")

        for r in self._compare_results:
            print(f"{r.op_name:<20} {'✓' if r.exact_match else '✗':^6} "
                  f"{r.max_abs_error:>10.2e} {r.cosine_sim:>10.6f} "
                  f"{r.qsnr_db:>8.1f} {r.status:^10}")

        print("=" * 80)
        print(f"Summary: {pass_count} PASS, {fail_count} FAIL (total: {len(self._compare_results)})")


@contextmanager
def golden_mode(
    golden: Literal["cpp", "python", "none"] = "python",
    compare: Literal["exact", "fuzzy", "quantized", "none"] = "none",
    quantize: Optional[str] = None,
    profile: bool = False,
    backward: bool = False,
    verbose: bool = False,
):
    """Context manager: 在此范围内使用 CPU golden

    Args:
        golden: Golden 模式 ("cpp" | "python" | "none")
        compare: 比对模式 ("exact" | "fuzzy" | "quantized" | "none")
        quantize: 量化类型 ("gfp16" | "gfp8" | "bfp16" | "bfp8" | None)
        profile: 是否启用 Paper Analysis
        backward: 是否启用反向传播 golden (训练模式)
        verbose: 是否打印详细信息

    Example:
        # 推理模式
        with golden_mode(golden="cpp", compare="fuzzy", quantize="gfp16"):
            y = model(x)

        # 训练模式 (前向+反向都走 golden)
        with golden_mode(golden="python", backward=True) as backend:
            y = model(x)
            loss = y.sum()
            loss.backward()  # 反向传播也走 golden
    """
    backend = TorchGoldenBackend(TorchBackendConfig(
        golden_mode=golden,
        compare_mode=compare,
        quantize_type=quantize,
        profile_enabled=profile,
        backward_enabled=backward,
        verbose=verbose,
    ))

    backend.clear()
    backend.enable()
    try:
        yield backend
    finally:
        backend.disable()
