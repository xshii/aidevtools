"""验证算子调用与 PyTorch API 兼容

注意: 此文件测试 API 兼容性，不测试数值一致性。
数值比对应在 torch_backend 劫持模式下进行。
"""
import numpy as np
import pytest

from aidevtools.ops import _functional as F
from aidevtools.ops.cpu_golden import is_cpu_golden_available, set_cpu_golden_dtype

torch = pytest.importorskip("torch")


class TestActivations:
    """激活函数 - 有 cpu_golden 实现"""

    def setup_method(self):
        """清理测试间的注册状态"""
        from aidevtools.ops.base import _golden_cpp_registry, clear
        _golden_cpp_registry.clear()
        clear()
        if is_cpu_golden_available():
            set_cpu_golden_dtype("gfp16")

    def test_relu_shape(self):
        if not is_cpu_golden_available():
            pytest.skip("CPU golden not available")
        x_np = np.random.randn(2, 3, 4).astype(np.float32)
        y = F.relu(x_np)
        assert y.shape == x_np.shape

    def test_gelu_shape(self):
        if not is_cpu_golden_available():
            pytest.skip("CPU golden not available")
        x_np = np.random.randn(2, 3, 4).astype(np.float32)
        y = F.gelu(x_np)
        assert y.shape == x_np.shape

    def test_sigmoid_shape(self):
        if not is_cpu_golden_available():
            pytest.skip("CPU golden not available")
        x_np = np.random.randn(2, 3, 4).astype(np.float32)
        y = F.sigmoid(x_np)
        assert y.shape == x_np.shape

    def test_tanh_shape(self):
        if not is_cpu_golden_available():
            pytest.skip("CPU golden not available")
        x_np = np.random.randn(2, 3, 4).astype(np.float32)
        y = F.tanh(x_np)
        assert y.shape == x_np.shape

    def test_silu_shape(self):
        if not is_cpu_golden_available():
            pytest.skip("CPU golden not available")
        x_np = np.random.randn(2, 3, 4).astype(np.float32)
        y = F.silu(x_np)
        assert y.shape == x_np.shape


class TestSoftmax:
    """Softmax 有 cpu_golden 实现"""

    def setup_method(self):
        if is_cpu_golden_available():
            set_cpu_golden_dtype("gfp16")

    def test_softmax_shape(self):
        if not is_cpu_golden_available():
            pytest.skip("CPU golden not available")

        x_np = np.random.randn(2, 3, 4).astype(np.float32)
        y_ours = F.softmax(x_np, dim=-1)

        assert y_ours.shape == x_np.shape
        # 每行和接近 1
        assert np.allclose(y_ours.sum(axis=-1), 1.0, atol=1e-2)


class TestLinear:
    """Linear 有 cpu_golden 实现"""

    def setup_method(self):
        if is_cpu_golden_available():
            set_cpu_golden_dtype("gfp16")

    def test_linear_no_bias_shape(self):
        if not is_cpu_golden_available():
            pytest.skip("CPU golden not available")

        x_np = np.random.randn(2, 3, 4).astype(np.float32)
        w_np = np.random.randn(8, 4).astype(np.float32)

        y_ours = F.linear(x_np, w_np)
        assert y_ours.shape == (2, 3, 8)

    def test_linear_with_bias_shape(self):
        if not is_cpu_golden_available():
            pytest.skip("CPU golden not available")

        x_np = np.random.randn(2, 3, 4).astype(np.float32)
        w_np = np.random.randn(8, 4).astype(np.float32)
        b_np = np.random.randn(8).astype(np.float32)

        y_ours = F.linear(x_np, w_np, b_np)
        assert y_ours.shape == (2, 3, 8)


class TestNormalization:
    """Normalization - LayerNorm 有 cpu_golden，BatchNorm 无"""

    def setup_method(self):
        if is_cpu_golden_available():
            set_cpu_golden_dtype("gfp16")

    def test_layernorm_shape(self):
        if not is_cpu_golden_available():
            pytest.skip("CPU golden not available")

        x_np = np.random.randn(2, 3, 4).astype(np.float32)
        weight_np = np.random.randn(4).astype(np.float32)
        bias_np = np.random.randn(4).astype(np.float32)

        y_ours = F.layernorm(x_np, normalized_shape=(4,), weight=weight_np, bias=bias_np, eps=1e-5)
        assert y_ours.shape == x_np.shape

    def test_batchnorm_not_implemented(self):
        x_np = np.random.randn(4, 8).astype(np.float32)
        gamma_np = np.ones(8, dtype=np.float32)
        beta_np = np.zeros(8, dtype=np.float32)

        with pytest.raises(NotImplementedError):
            F.batchnorm(x_np, gamma_np, beta_np)


class TestMatMul:
    """MatMul 有 cpu_golden 实现"""

    def setup_method(self):
        if is_cpu_golden_available():
            set_cpu_golden_dtype("gfp16")

    def test_matmul_2d_shape(self):
        if not is_cpu_golden_available():
            pytest.skip("CPU golden not available")

        a_np = np.random.randn(3, 4).astype(np.float32)
        b_np = np.random.randn(4, 5).astype(np.float32)

        y_ours = F.matmul(a_np, b_np)
        assert y_ours.shape == (3, 5)

    def test_matmul_batch_shape(self):
        if not is_cpu_golden_available():
            pytest.skip("CPU golden not available")

        a_np = np.random.randn(2, 3, 4).astype(np.float32)
        b_np = np.random.randn(2, 4, 5).astype(np.float32)

        y_ours = F.matmul(a_np, b_np)
        assert y_ours.shape == (2, 3, 5)


class TestAttention:
    """Attention - 无 cpu_golden 实现"""

    def test_attention_not_implemented(self):
        batch, heads, seq, dim = 2, 4, 8, 16
        q_np = np.random.randn(batch, heads, seq, dim).astype(np.float32)
        k_np = np.random.randn(batch, heads, seq, dim).astype(np.float32)
        v_np = np.random.randn(batch, heads, seq, dim).astype(np.float32)

        with pytest.raises(NotImplementedError):
            F.attention(q_np, k_np, v_np)


class TestLossFunctionsNotImplemented:
    """损失函数 - 无 cpu_golden 实现"""

    def test_cross_entropy_not_implemented(self):
        logits_np = np.random.randn(4, 10).astype(np.float32)
        target_np = np.array([1, 3, 5, 7], dtype=np.int64)

        with pytest.raises(NotImplementedError):
            F.cross_entropy(logits_np, target_np, reduction="mean")

    def test_mse_loss_not_implemented(self):
        pred_np = np.random.randn(4, 10).astype(np.float32)
        target_np = np.random.randn(4, 10).astype(np.float32)

        with pytest.raises(NotImplementedError):
            F.mse_loss(pred_np, target_np, reduction="mean")

    def test_l1_loss_not_implemented(self):
        pred_np = np.random.randn(4, 10).astype(np.float32)
        target_np = np.random.randn(4, 10).astype(np.float32)

        with pytest.raises(NotImplementedError):
            F.l1_loss(pred_np, target_np, reduction="mean")

    def test_smooth_l1_loss_not_implemented(self):
        pred_np = np.random.randn(4, 10).astype(np.float32)
        target_np = np.random.randn(4, 10).astype(np.float32)

        with pytest.raises(NotImplementedError):
            F.smooth_l1_loss(pred_np, target_np, reduction="mean")

    def test_bce_with_logits_not_implemented(self):
        logits_np = np.random.randn(4, 10).astype(np.float32)
        target_np = np.random.rand(4, 10).astype(np.float32)

        with pytest.raises(NotImplementedError):
            F.bce_with_logits(logits_np, target_np, reduction="mean")


class TestElementwiseOps:
    """逐元素运算 - 有 cpu_golden 实现"""

    def setup_method(self):
        if is_cpu_golden_available():
            set_cpu_golden_dtype("gfp16")

    def test_add_shape(self):
        if not is_cpu_golden_available():
            pytest.skip("CPU golden not available")
        a_np = np.random.randn(2, 3, 4).astype(np.float32)
        b_np = np.random.randn(2, 3, 4).astype(np.float32)
        c = F.add(a_np, b_np)
        assert c.shape == a_np.shape

    def test_mul_shape(self):
        if not is_cpu_golden_available():
            pytest.skip("CPU golden not available")
        a_np = np.random.randn(2, 3, 4).astype(np.float32)
        b_np = np.random.randn(2, 3, 4).astype(np.float32)
        c = F.mul(a_np, b_np)
        assert c.shape == a_np.shape

    def test_div_shape(self):
        if not is_cpu_golden_available():
            pytest.skip("CPU golden not available")
        a_np = np.random.randn(2, 3, 4).astype(np.float32)
        b_np = np.random.randn(2, 3, 4).astype(np.float32) + 0.1
        c = F.div(a_np, b_np)
        assert c.shape == a_np.shape


class TestTranspose:
    """Transpose 有 cpu_golden 实现"""

    def setup_method(self):
        if is_cpu_golden_available():
            set_cpu_golden_dtype("gfp16")

    def test_transpose_2d_shape(self):
        if not is_cpu_golden_available():
            pytest.skip("CPU golden not available")

        x_np = np.random.randn(3, 4).astype(np.float32)
        y_ours = F.transpose(x_np)
        assert y_ours.shape == (4, 3)

    def test_transpose_4d_shape(self):
        if not is_cpu_golden_available():
            pytest.skip("CPU golden not available")

        x_np = np.random.randn(2, 4, 8, 16).astype(np.float32)
        y_ours = F.transpose(x_np, axes=(0, 1, 3, 2))
        assert y_ours.shape == (2, 4, 16, 8)
