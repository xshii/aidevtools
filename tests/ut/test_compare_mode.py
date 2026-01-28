"""比对模式单元测试"""
import numpy as np
import pytest

from aidevtools import ops
from aidevtools.ops import _functional as F
from aidevtools.ops.base import CompareMode


class TestCompareMode:
    """CompareMode 枚举测试"""

    def test_enum_values(self):
        """枚举值"""
        assert CompareMode.SINGLE_OP.value == "single_op"
        assert CompareMode.FULL_GRAPH.value == "full_graph"
        assert CompareMode.MIXED.value == "mixed"


class TestCompareModeAPI:
    """比对模式 API 测试"""

    def setup_method(self):
        ops.clear()
        ops.set_compare_mode(CompareMode.SINGLE_OP)

    def test_set_get_compare_mode(self):
        """设置和获取比对模式"""
        ops.set_compare_mode(CompareMode.FULL_GRAPH)
        assert ops.get_compare_mode() == CompareMode.FULL_GRAPH

        ops.set_compare_mode(CompareMode.MIXED)
        assert ops.get_compare_mode() == CompareMode.MIXED

    def test_default_mode(self):
        """默认模式是 SINGLE_OP"""
        ops.clear()
        assert ops.get_compare_mode() == CompareMode.SINGLE_OP


class TestSingleOpMode:
    """单算子模式测试"""

    def setup_method(self):
        from aidevtools.ops.cpu_golden import is_cpu_golden_available, set_cpu_golden_dtype

        if not is_cpu_golden_available():
            pytest.skip("CPU golden not available")

        set_cpu_golden_dtype("gfp16")
        ops.clear()
        ops.set_compare_mode(CompareMode.SINGLE_OP)

    def test_each_op_compared(self):
        """每个算子都比对"""
        x = np.random.randn(4, 8).astype(np.float32)

        y = F.relu(x)
        y = F.gelu(y)

        records = ops.get_records()
        assert len(records) == 2

        # 每个算子都有 reference
        for r in records:
            assert r["golden"] is not None
            assert r["reference"] is not None


class TestFullGraphMode:
    """完整图模式测试"""

    def setup_method(self):
        from aidevtools.ops.cpu_golden import is_cpu_golden_available, set_cpu_golden_dtype

        if not is_cpu_golden_available():
            pytest.skip("CPU golden not available")

        set_cpu_golden_dtype("gfp16")
        ops.clear()
        ops.set_compare_mode(CompareMode.FULL_GRAPH)

    def test_graph_recorded(self):
        """计算图被记录"""
        x = np.random.randn(4, 8).astype(np.float32)

        y = F.relu(x)
        y = F.gelu(y)

        graph = ops.get_graph()
        assert len(graph) == 2
        assert "relu_0" in graph
        assert "gelu_0" in graph

    def test_no_reference_computed(self):
        """不计算 reference（除非标记）"""
        x = np.random.randn(4, 8).astype(np.float32)

        y = F.relu(x)
        y = F.gelu(y)

        records = ops.get_records()
        assert len(records) == 2

        # reference 为 None（未标记比对点）
        for r in records:
            assert r["golden"] is not None
            assert r["reference"] is None

    def test_mark_compare_point(self):
        """标记比对点"""
        x = np.random.randn(4, 8).astype(np.float32)

        ops.mark_compare_point("gelu_0")

        y = F.relu(x)
        y = F.gelu(y)

        records = ops.get_records()
        assert len(records) == 2

        # relu_0 没有 reference
        assert records[0]["reference"] is None
        # gelu_0 有 reference（标记了比对点）
        assert records[1]["reference"] is not None


class TestMixedMode:
    """混合模式测试"""

    def setup_method(self):
        from aidevtools.ops.cpu_golden import is_cpu_golden_available, set_cpu_golden_dtype

        if not is_cpu_golden_available():
            pytest.skip("CPU golden not available")

        set_cpu_golden_dtype("gfp16")
        ops.clear()
        ops.set_compare_mode(CompareMode.MIXED)

    def test_graph_recorded(self):
        """计算图被记录"""
        x = np.random.randn(4, 8).astype(np.float32)

        y = F.relu(x)
        y = F.gelu(y)

        graph = ops.get_graph()
        assert len(graph) == 2

    def test_generate_test_cases(self):
        """生成测试用例"""
        x = np.random.randn(4, 8).astype(np.float32)

        y = F.relu(x)
        y = F.gelu(y)

        test_cases = ops.generate_test_cases()

        # 2 个单算子 + 1 个双算子链 + 1 个完整图 = 4
        assert len(test_cases) >= 3

        # 检查测试类型
        types = {tc.test_type for tc in test_cases}
        assert "standalone" in types
        assert "full_graph" in types


class TestTracedTensorIntegration:
    """TracedTensor 集成测试"""

    def setup_method(self):
        from aidevtools.ops.cpu_golden import is_cpu_golden_available, set_cpu_golden_dtype

        if not is_cpu_golden_available():
            pytest.skip("CPU golden not available")

        set_cpu_golden_dtype("gfp16")
        ops.clear()
        ops.set_compare_mode(CompareMode.SINGLE_OP)

    def test_traced_input_returns_traced_output(self):
        """TracedTensor 输入返回 TracedTensor 输出"""
        x = ops.traced(np.random.randn(4, 8).astype(np.float32), "gfp16")

        y = F.relu(x)

        assert isinstance(y, ops.TracedTensor)
        assert y.source_op == "relu_0"
        assert y.dtype == "gfp16"

    def test_numpy_input_returns_numpy_output(self):
        """numpy 输入返回 numpy 输出（兼容）"""
        x = np.random.randn(4, 8).astype(np.float32)

        y = F.relu(x)

        assert isinstance(y, np.ndarray)
        assert not isinstance(y, ops.TracedTensor)

    def test_chain_traced_tensors(self):
        """链式调用 TracedTensor"""
        x = ops.traced(np.random.randn(4, 8).astype(np.float32), "gfp16")

        y = F.relu(x)
        z = F.gelu(y)

        assert isinstance(z, ops.TracedTensor)
        assert z.source_op == "gelu_0"

        # 检查计算结果正确
        result = z.numpy()
        assert result.shape == (4, 8)


class TestGraphOps:
    """计算图操作测试"""

    def setup_method(self):
        from aidevtools.ops.cpu_golden import is_cpu_golden_available, set_cpu_golden_dtype

        if not is_cpu_golden_available():
            pytest.skip("CPU golden not available")

        set_cpu_golden_dtype("gfp16")
        ops.clear()
        ops.set_compare_mode(CompareMode.MIXED)

    def test_get_graph_ops(self):
        """获取算子列表"""
        x = np.random.randn(4, 8).astype(np.float32)

        F.relu(x)
        F.gelu(x)
        F.sigmoid(x)

        op_names = ops.get_graph_ops()
        assert op_names == ["relu_0", "gelu_0", "sigmoid_0"]

    def test_graph_node_inputs(self):
        """图节点记录输入"""
        x = ops.traced(np.random.randn(4, 8).astype(np.float32), "gfp16")

        y = F.relu(x)
        z = F.gelu(y)

        graph = ops.get_graph()

        # gelu_0 的输入来自 relu_0
        assert "relu_0" in graph["gelu_0"].inputs

    def test_clear_resets_graph(self):
        """clear 清空计算图"""
        x = np.random.randn(4, 8).astype(np.float32)

        F.relu(x)
        assert len(ops.get_graph()) == 1

        ops.clear()
        assert len(ops.get_graph()) == 0


class TestTestGenerator:
    """测试生成器测试"""

    def setup_method(self):
        from aidevtools.ops.cpu_golden import is_cpu_golden_available, set_cpu_golden_dtype

        if not is_cpu_golden_available():
            pytest.skip("CPU golden not available")

        set_cpu_golden_dtype("gfp16")
        ops.clear()
        ops.set_compare_mode(CompareMode.MIXED)

    def test_generate_standalone_only(self):
        """只生成单算子测试"""
        x = np.random.randn(4, 8).astype(np.float32)

        F.relu(x)
        F.gelu(x)

        test_cases = ops.generate_test_cases(
            include_standalone=True,
            include_chain=False,
            include_full_graph=False,
        )

        assert len(test_cases) == 2
        for tc in test_cases:
            assert tc.test_type == "standalone"

    def test_generate_full_graph_only(self):
        """只生成完整图测试"""
        x = np.random.randn(4, 8).astype(np.float32)

        F.relu(x)
        F.gelu(x)

        test_cases = ops.generate_test_cases(
            include_standalone=False,
            include_chain=False,
            include_full_graph=True,
        )

        assert len(test_cases) == 1
        assert test_cases[0].test_type == "full_graph"

    def test_empty_graph(self):
        """空图返回空列表"""
        test_cases = ops.generate_test_cases()
        assert len(test_cases) == 0
