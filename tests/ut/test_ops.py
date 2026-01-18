"""算子 API 单元测试"""
import pytest
import numpy as np
from pathlib import Path


class TestOpsBase:
    """算子基础框架测试"""

    def setup_method(self):
        from aidevtools.ops.base import clear, _golden_registry
        clear()
        _golden_registry.clear()

    def test_register_golden(self):
        """注册 Golden 实现"""
        from aidevtools.ops.base import register_golden, has_golden

        @register_golden("test_op")
        def golden_test(x):
            return x * 2

        assert has_golden("test_op")
        assert not has_golden("unknown_op")

    def test_op_without_golden(self):
        """未注册 golden 时只执行 reference"""
        from aidevtools.ops.base import clear, get_records
        from aidevtools.ops.nn import relu

        clear()
        x = np.array([[-1, 0, 1], [2, -2, 3]], dtype=np.float32)
        y = relu(x)

        assert np.allclose(y, np.array([[0, 0, 1], [2, 0, 3]]))
        records = get_records()
        assert len(records) == 1
        assert records[0]["name"] == "relu_0"
        assert records[0]["golden"] is not None
        assert records[0]["result"] is None  # 未注册 golden

    def test_op_with_golden(self):
        """注册 golden 后同时执行两种实现"""
        from aidevtools.ops.base import register_golden, clear, get_records
        from aidevtools.ops.nn import ReLU

        @register_golden("relu")
        def golden_relu(x):
            return np.maximum(0, x) + 0.001  # 故意加点误差

        clear()
        relu = ReLU()
        x = np.array([[-1, 0, 1]], dtype=np.float32)
        y = relu(x)

        # 返回 reference 结果
        assert np.allclose(y, np.array([[0, 0, 1]]))

        records = get_records()
        assert len(records) == 1
        assert records[0]["golden"] is not None
        assert records[0]["result"] is not None
        # golden 和 result 应该有微小差异
        assert not np.allclose(records[0]["golden"], records[0]["result"])


class TestNNOps:
    """神经网络算子测试"""

    def setup_method(self):
        from aidevtools.ops.base import clear
        clear()

    def test_linear(self):
        """Linear 算子"""
        from aidevtools.ops.nn import linear

        x = np.random.randn(2, 3, 4).astype(np.float32)
        w = np.random.randn(4, 8).astype(np.float32)
        b = np.random.randn(8).astype(np.float32)

        y = linear(x, w, b)
        expected = np.matmul(x, w) + b
        assert np.allclose(y, expected)

    def test_relu(self):
        """ReLU 算子"""
        from aidevtools.ops.nn import relu

        x = np.array([-2, -1, 0, 1, 2], dtype=np.float32)
        y = relu(x)
        assert np.allclose(y, [0, 0, 0, 1, 2])

    def test_gelu(self):
        """GELU 算子"""
        from aidevtools.ops.nn import gelu

        x = np.array([0], dtype=np.float32)
        y = gelu(x)
        assert np.allclose(y, [0], atol=1e-5)

    def test_softmax(self):
        """Softmax 算子"""
        from aidevtools.ops.nn import softmax

        x = np.array([[1, 2, 3], [1, 1, 1]], dtype=np.float32)
        y = softmax(x)
        # 每行和为 1
        assert np.allclose(y.sum(axis=-1), [1, 1])

    def test_layernorm(self):
        """LayerNorm 算子"""
        from aidevtools.ops.nn import layernorm

        x = np.random.randn(2, 3, 4).astype(np.float32)
        gamma = np.ones(4, dtype=np.float32)
        beta = np.zeros(4, dtype=np.float32)

        y = layernorm(x, gamma, beta)
        # 归一化后均值接近 0，方差接近 1
        assert np.allclose(y.mean(axis=-1), 0, atol=1e-5)
        assert np.allclose(y.var(axis=-1), 1, atol=1e-3)  # float32 精度限制

    def test_attention(self):
        """Attention 算子"""
        from aidevtools.ops.nn import attention

        batch, heads, seq, dim = 2, 4, 8, 16
        q = np.random.randn(batch, heads, seq, dim).astype(np.float32)
        k = np.random.randn(batch, heads, seq, dim).astype(np.float32)
        v = np.random.randn(batch, heads, seq, dim).astype(np.float32)

        y = attention(q, k, v)
        assert y.shape == (batch, heads, seq, dim)

    def test_sigmoid(self):
        """Sigmoid 算子"""
        from aidevtools.ops.nn import sigmoid

        x = np.array([0, 1, -1], dtype=np.float32)
        y = sigmoid(x)
        expected = 1 / (1 + np.exp(-x))
        assert np.allclose(y, expected)
        assert np.allclose(y[0], 0.5)  # sigmoid(0) = 0.5

    def test_tanh(self):
        """Tanh 算子"""
        from aidevtools.ops.nn import tanh

        x = np.array([0, 1, -1], dtype=np.float32)
        y = tanh(x)
        assert np.allclose(y, np.tanh(x))
        assert np.allclose(y[0], 0)  # tanh(0) = 0

    def test_batchnorm(self):
        """BatchNorm 算子"""
        from aidevtools.ops.nn import batchnorm

        x = np.random.randn(4, 8).astype(np.float32)
        gamma = np.ones(8, dtype=np.float32)
        beta = np.zeros(8, dtype=np.float32)

        y = batchnorm(x, gamma, beta)
        # 归一化后每列均值接近 0
        assert np.allclose(y.mean(axis=0), 0, atol=1e-5)

    def test_batchnorm_with_stats(self):
        """BatchNorm 使用预计算统计量"""
        from aidevtools.ops.nn import batchnorm

        x = np.random.randn(4, 8).astype(np.float32)
        gamma = np.ones(8, dtype=np.float32)
        beta = np.zeros(8, dtype=np.float32)
        mean = np.zeros(8, dtype=np.float32)
        var = np.ones(8, dtype=np.float32)

        y = batchnorm(x, gamma, beta, mean=mean, var=var)
        # 使用固定统计量时，输出应该接近输入
        assert np.allclose(y, x, atol=1e-5)

    def test_embedding(self):
        """Embedding 算子"""
        from aidevtools.ops.nn import embedding

        vocab_size, embed_dim = 100, 64
        embed_table = np.random.randn(vocab_size, embed_dim).astype(np.float32)
        input_ids = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int64)

        y = embedding(input_ids, embed_table)
        assert y.shape == (2, 3, 64)
        assert np.allclose(y[0, 0], embed_table[1])
        assert np.allclose(y[1, 2], embed_table[6])

    def test_matmul(self):
        """MatMul 算子"""
        from aidevtools.ops.nn import matmul

        a = np.random.randn(2, 3, 4).astype(np.float32)
        b = np.random.randn(2, 4, 5).astype(np.float32)

        y = matmul(a, b)
        assert y.shape == (2, 3, 5)
        assert np.allclose(y, np.matmul(a, b))

    def test_add(self):
        """Add 算子"""
        from aidevtools.ops.nn import add

        a = np.array([1, 2, 3], dtype=np.float32)
        b = np.array([4, 5, 6], dtype=np.float32)

        y = add(a, b)
        assert np.allclose(y, [5, 7, 9])

    def test_mul(self):
        """Mul 算子"""
        from aidevtools.ops.nn import mul

        a = np.array([1, 2, 3], dtype=np.float32)
        b = np.array([4, 5, 6], dtype=np.float32)

        y = mul(a, b)
        assert np.allclose(y, [4, 10, 18])

    def test_div(self):
        """Div 算子"""
        from aidevtools.ops.nn import div

        a = np.array([4, 10, 18], dtype=np.float32)
        b = np.array([2, 5, 6], dtype=np.float32)

        y = div(a, b)
        assert np.allclose(y, [2, 2, 3])

    def test_op_repr(self):
        """Op.__repr__ 测试"""
        from aidevtools.ops.nn import linear, relu
        from aidevtools.ops.base import register_golden

        # 未注册 golden
        assert "linear" in repr(linear)
        assert "✗" in repr(linear)

        # 注册 golden 后
        @register_golden("relu")
        def golden_relu(x):
            return np.maximum(0, x)

        assert "✓" in repr(relu)


class TestOpsDump:
    """算子数据导出测试"""

    def setup_method(self):
        from aidevtools.ops.base import clear
        clear()

    def test_dump(self, tmp_path):
        """导出数据"""
        from aidevtools.ops.base import clear, dump
        from aidevtools.ops.nn import linear, relu

        clear()
        x = np.random.randn(2, 4).astype(np.float32)
        w = np.random.randn(4, 8).astype(np.float32)

        y = linear(x, w)
        y = relu(y)

        dump(str(tmp_path))

        assert (tmp_path / "linear_0_golden.bin").exists()
        assert (tmp_path / "linear_0_input.bin").exists()
        assert (tmp_path / "linear_0_weight.bin").exists()
        assert (tmp_path / "relu_0_golden.bin").exists()

    def test_gen_csv(self, tmp_path):
        """生成 CSV"""
        from aidevtools.ops.base import clear, dump, gen_csv
        from aidevtools.ops.nn import linear

        clear()
        x = np.random.randn(2, 4).astype(np.float32)
        w = np.random.randn(4, 8).astype(np.float32)

        linear(x, w)
        dump(str(tmp_path))
        csv_path = gen_csv(str(tmp_path), "test_model")

        assert Path(csv_path).exists()
        content = Path(csv_path).read_text()
        assert "linear_0" in content
        assert "golden_bin" in content
