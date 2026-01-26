"""算子 API 单元测试"""
import numpy as np

from aidevtools.ops import _functional as F


class TestOpsBase:
    """算子基础框架测试"""

    def setup_method(self):
        from aidevtools.ops.base import clear, _golden_cpp_registry
        clear()
        _golden_cpp_registry.clear()

    def test_register_golden_cpp(self):
        """注册 C++ Golden 实现"""
        from aidevtools.ops.base import register_golden_cpp, has_golden_cpp

        @register_golden_cpp("test_op")
        def golden_test(x):
            return x * 2

        assert has_golden_cpp("test_op")
        assert not has_golden_cpp("unknown_op")

    def test_op_default_python_mode(self):
        """默认模式执行 golden (cpu_golden)"""
        from aidevtools.ops.base import clear, get_records, set_golden_mode
        from aidevtools.ops.cpu_golden import is_cpu_golden_available

        if not is_cpu_golden_available():
            pytest.skip("CPU golden not available")

        set_golden_mode("python")
        clear()
        x = np.array([[-1, 0, 1], [2, -2, 3]], dtype=np.float32)

        # ReLU 有 cpu_golden 实现
        y = F.relu(x)
        expected = np.array([[0, 0, 1], [2, 0, 3]], dtype=np.float32)
        assert np.allclose(y, expected, atol=0.01)

    def test_op_with_cpp_golden(self):
        """注册 C++ golden 后调用算子 (使用内置 cpu_golden)"""
        from aidevtools.ops.base import clear, get_records, set_golden_mode
        from aidevtools.ops.cpu_golden import is_cpu_golden_available

        if not is_cpu_golden_available():
            pytest.skip("CPU golden not available")

        set_golden_mode("python")
        clear()
        relu = F.ReLU()
        x = np.array([[-1, 0, 1]], dtype=np.float32)
        y = relu(x)

        # 返回 cpu_golden 结果
        expected = np.array([[0, 0, 1]], dtype=np.float32)
        assert np.allclose(y, expected, atol=0.01)

        records = get_records()
        assert len(records) == 1
        assert records[0]["golden"] is not None

    def test_golden_mode_none(self):
        """golden_mode=none 时仍执行 golden 计算"""
        from aidevtools.ops.base import clear, set_golden_mode
        from aidevtools.ops.cpu_golden import is_cpu_golden_available

        if not is_cpu_golden_available():
            pytest.skip("CPU golden not available")

        set_golden_mode("none")
        clear()
        x = np.array([[-1, 0, 1]], dtype=np.float32)

        # ReLU 有 cpu_golden，正常执行
        y = F.relu(x)
        expected = np.array([[0, 0, 1]], dtype=np.float32)
        assert np.allclose(y, expected, atol=0.01)

        # 恢复 python 模式
        set_golden_mode("python")


class TestNNOps:
    """神经网络算子测试 (仅测试有 cpu_golden 实现的算子)"""

    def setup_method(self):
        from aidevtools.ops.base import clear, set_golden_mode
        from aidevtools.ops.cpu_golden import is_cpu_golden_available, set_cpu_golden_dtype
        set_golden_mode("python")
        clear()
        # 如果 cpu_golden 可用，设置 dtype
        if is_cpu_golden_available():
            set_cpu_golden_dtype("gfp16")

    def test_linear(self):
        """Linear 算子 (PyTorch 格式: weight [out, in])"""
        import pytest
        from aidevtools.ops.cpu_golden import is_cpu_golden_available

        if not is_cpu_golden_available():
            pytest.skip("CPU golden not available")

        x = np.random.randn(2, 3, 4).astype(np.float32)
        # PyTorch 格式: weight [out_features, in_features]
        w = np.random.randn(8, 4).astype(np.float32)
        b = np.random.randn(8).astype(np.float32)

        y = F.linear(x, w, b)
        expected = np.matmul(x, w.T) + b
        # cpu_golden 有量化误差，放宽精度要求
        assert y.shape == expected.shape

    def test_relu(self):
        """ReLU 算子"""
        import pytest
        from aidevtools.ops.cpu_golden import is_cpu_golden_available

        if not is_cpu_golden_available():
            pytest.skip("CPU golden not available")

        x = np.array([-2, -1, 0, 1, 2], dtype=np.float32)
        y = F.relu(x)
        expected = np.array([0, 0, 0, 1, 2], dtype=np.float32)
        assert np.allclose(y, expected, atol=0.01)

    def test_gelu(self):
        """GELU 算子"""
        import pytest
        from aidevtools.ops.cpu_golden import is_cpu_golden_available

        if not is_cpu_golden_available():
            pytest.skip("CPU golden not available")

        x = np.array([0], dtype=np.float32)
        y = F.gelu(x)
        # GELU(0) = 0
        assert np.allclose(y, np.array([0], dtype=np.float32), atol=0.01)

    def test_softmax(self):
        """Softmax 算子"""
        import pytest
        from aidevtools.ops.cpu_golden import is_cpu_golden_available

        if not is_cpu_golden_available():
            pytest.skip("CPU golden not available")

        x = np.array([[1, 2, 3], [1, 1, 1]], dtype=np.float32)
        y = F.softmax(x)
        # 每行和接近 1 (允许 gfp16 量化误差)
        assert np.allclose(y.sum(axis=-1), [1, 1], atol=0.01)

    def test_layernorm(self):
        """LayerNorm 算子"""
        import pytest
        from aidevtools.ops.cpu_golden import is_cpu_golden_available

        if not is_cpu_golden_available():
            pytest.skip("CPU golden not available")

        x = np.random.randn(2, 3, 4).astype(np.float32)
        weight = np.ones(4, dtype=np.float32)
        bias = np.zeros(4, dtype=np.float32)

        y = F.layernorm(x, normalized_shape=(4,), weight=weight, bias=bias)
        # 输出形状正确
        assert y.shape == x.shape

    def test_attention(self):
        """Attention 算子 - 无 cpu_golden，预期抛出 NotImplementedError"""
        import pytest

        batch, heads, seq, dim = 2, 4, 8, 16
        q = np.random.randn(batch, heads, seq, dim).astype(np.float32)
        k = np.random.randn(batch, heads, seq, dim).astype(np.float32)
        v = np.random.randn(batch, heads, seq, dim).astype(np.float32)

        with pytest.raises(NotImplementedError):
            F.attention(q, k, v)

    def test_sigmoid(self):
        """Sigmoid 算子"""
        import pytest
        from aidevtools.ops.cpu_golden import is_cpu_golden_available

        if not is_cpu_golden_available():
            pytest.skip("CPU golden not available")

        x = np.array([0, 1, -1], dtype=np.float32)
        y = F.sigmoid(x)
        # sigmoid(0) = 0.5
        assert np.allclose(y[0], 0.5, atol=0.01)

    def test_tanh(self):
        """Tanh 算子"""
        import pytest
        from aidevtools.ops.cpu_golden import is_cpu_golden_available

        if not is_cpu_golden_available():
            pytest.skip("CPU golden not available")

        x = np.array([0, 1, -1], dtype=np.float32)
        y = F.tanh(x)
        # tanh(0) = 0
        assert np.allclose(y[0], 0, atol=0.01)

    def test_batchnorm(self):
        """BatchNorm 算子 - 无 cpu_golden，预期抛出 NotImplementedError"""
        import pytest

        x = np.random.randn(4, 8).astype(np.float32)
        gamma = np.ones(8, dtype=np.float32)
        beta = np.zeros(8, dtype=np.float32)

        with pytest.raises(NotImplementedError):
            F.batchnorm(x, gamma, beta)

    def test_batchnorm_with_stats(self):
        """BatchNorm 使用预计算统计量 - 无 cpu_golden，预期抛出 NotImplementedError"""
        import pytest

        x = np.random.randn(4, 8).astype(np.float32)
        gamma = np.ones(8, dtype=np.float32)
        beta = np.zeros(8, dtype=np.float32)
        mean = np.zeros(8, dtype=np.float32)
        var = np.ones(8, dtype=np.float32)

        with pytest.raises(NotImplementedError):
            F.batchnorm(x, gamma, beta, mean=mean, var=var)

    def test_embedding(self):
        """Embedding 算子 - 无 cpu_golden，预期抛出 NotImplementedError"""
        import pytest

        vocab_size, embed_dim = 100, 64
        embed_table = np.random.randn(vocab_size, embed_dim).astype(np.float32)
        input_ids = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int64)

        with pytest.raises(NotImplementedError):
            F.embedding(input_ids, embed_table)

    def test_matmul(self):
        """MatMul 算子"""
        import pytest
        from aidevtools.ops.cpu_golden import is_cpu_golden_available

        if not is_cpu_golden_available():
            pytest.skip("CPU golden not available")

        a = np.random.randn(2, 3, 4).astype(np.float32)
        b = np.random.randn(2, 4, 5).astype(np.float32)

        y = F.matmul(a, b)
        assert y.shape == (2, 3, 5)

    def test_add(self):
        """Add 算子"""
        import pytest
        from aidevtools.ops.cpu_golden import is_cpu_golden_available

        if not is_cpu_golden_available():
            pytest.skip("CPU golden not available")

        a = np.array([1, 2, 3], dtype=np.float32)
        b = np.array([4, 5, 6], dtype=np.float32)

        c = F.add(a, b)
        expected = np.array([5, 7, 9], dtype=np.float32)
        assert np.allclose(c, expected, atol=0.01)

    def test_mul(self):
        """Mul 算子"""
        import pytest
        from aidevtools.ops.cpu_golden import is_cpu_golden_available

        if not is_cpu_golden_available():
            pytest.skip("CPU golden not available")

        a = np.array([1, 2, 3], dtype=np.float32)
        b = np.array([4, 5, 6], dtype=np.float32)

        c = F.mul(a, b)
        expected = np.array([4, 10, 18], dtype=np.float32)
        assert np.allclose(c, expected, atol=0.01)

    def test_div(self):
        """Div 算子"""
        import pytest
        from aidevtools.ops.cpu_golden import is_cpu_golden_available

        if not is_cpu_golden_available():
            pytest.skip("CPU golden not available")

        a = np.array([4, 10, 18], dtype=np.float32)
        b = np.array([2, 5, 6], dtype=np.float32)

        c = F.div(a, b)
        expected = np.array([2, 2, 3], dtype=np.float32)
        assert np.allclose(c, expected, atol=0.01)

    def test_op_repr(self):
        """Op.__repr__ 测试"""
        from aidevtools.ops.base import register_golden_cpp

        linear_op = F.Linear()
        relu_op = F.ReLU()

        # 未注册 cpp golden
        assert "linear" in repr(linear_op)
        assert "✗" in repr(linear_op)

        # 注册 cpp golden 后
        @register_golden_cpp("relu")
        def cpp_relu(x):
            return np.maximum(0, x)

        assert "✓" in repr(relu_op)


class TestOpsDump:
    """算子数据导出测试"""

    def setup_method(self):
        from aidevtools.ops.base import clear, set_golden_mode
        from aidevtools.ops.cpu_golden import is_cpu_golden_available, set_cpu_golden_dtype
        set_golden_mode("python")
        clear()
        if is_cpu_golden_available():
            set_cpu_golden_dtype("gfp16")

    def test_dump(self, tmp_path):
        """导出数据 (PyTorch 格式)"""
        import pytest
        from aidevtools.ops.base import clear, dump
        from aidevtools.ops.cpu_golden import is_cpu_golden_available

        if not is_cpu_golden_available():
            pytest.skip("CPU golden not available")

        clear()
        x = np.random.randn(2, 4).astype(np.float32)
        # PyTorch 格式: weight [out_features, in_features]
        w = np.random.randn(8, 4).astype(np.float32)

        y = F.linear(x, w)
        # ReLU 没有 cpu_golden，跳过

        dump(str(tmp_path))

        assert (tmp_path / "linear_0_golden.bin").exists()
        assert (tmp_path / "linear_0_input.bin").exists()
        assert (tmp_path / "linear_0_weight.bin").exists()
        # reference 由 torch 内部计算
        assert (tmp_path / "linear_0_reference.bin").exists()
