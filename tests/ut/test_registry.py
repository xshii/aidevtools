"""测试 ops.registry 统一算子注册"""
from aidevtools.ops.registry import (
    get_op_meta,
    get_op_info,
    list_ops,
    validate_op,
    get_op_instance,
    OpMeta,
)


class TestRegisterOp:
    """测试 @register_op 装饰器"""

    def test_decorator_registers_op(self):
        """装饰器能正确注册算子"""
        # linear 已在 nn.py 中注册
        meta = get_op_meta("linear")
        assert meta is not None
        assert meta.name == "linear"
        assert "input" in meta.inputs
        assert "weight" in meta.inputs

    def test_decorator_with_optional(self):
        """装饰器能正确处理可选参数"""
        meta = get_op_meta("layernorm")
        assert meta is not None
        assert "eps" in meta.optional

    def test_decorator_with_cpp_golden(self):
        """装饰器能正确标记 cpp golden"""
        meta = get_op_meta("matmul")
        assert meta is not None
        assert meta.has_cpp_golden is True

        # relu 现在也有 cpp golden
        meta = get_op_meta("relu")
        assert meta is not None
        assert meta.has_cpp_golden is True

        # batchnorm 没有 cpp golden
        meta = get_op_meta("batchnorm")
        assert meta is not None
        assert meta.has_cpp_golden is False


class TestRegistryAPI:
    """测试注册表 API"""

    def test_list_ops(self):
        """列出所有算子"""
        ops = list_ops()
        assert "linear" in ops
        assert "matmul" in ops
        assert "softmax" in ops
        assert "layernorm" in ops
        assert "attention" in ops

    def test_validate_op(self):
        """验证算子是否存在"""
        assert validate_op("linear") is True
        assert validate_op("nonexistent") is False

    def test_get_op_info(self):
        """获取算子信息 (兼容旧 API)"""
        info = get_op_info("matmul")
        assert "inputs" in info
        assert "optional" in info
        assert "description" in info
        assert info["inputs"] == ["a", "b"]

    def test_get_op_info_unknown(self):
        """获取未知算子返回默认值"""
        info = get_op_info("unknown_op")
        assert info["inputs"] == ["x"]
        assert info["optional"] == []

    def test_get_op_instance(self):
        """获取算子实例"""
        instance = get_op_instance("linear")
        assert instance is not None
        assert instance.name == "linear"

        # 再次获取应返回相同实例
        instance2 = get_op_instance("linear")
        assert instance is instance2


class TestOpMeta:
    """测试 OpMeta 数据类"""

    def test_op_meta_defaults(self):
        """OpMeta 默认值"""
        meta = OpMeta(name="test")
        assert meta.inputs == ["x"]
        assert meta.optional == []
        assert meta.description == ""
        assert meta.has_cpp_golden is False
        assert meta.op_class is None


class TestXlsxOpRegistry:
    """测试 xlsx.op_registry 兼容性"""

    def test_get_default_ops(self):
        """获取默认算子列表"""
        from aidevtools.xlsx.op_registry import get_default_ops
        ops = get_default_ops()

        # 检查已注册的算子
        assert "linear" in ops
        assert "matmul" in ops

        # 检查额外算子
        assert "conv2d" in ops
        assert "pooling" in ops

    def test_backward_compat(self):
        """向后兼容性"""
        from aidevtools.xlsx.op_registry import (
            get_op_info as xlsx_get_op_info,
            list_ops as xlsx_list_ops,
            validate_op as xlsx_validate_op,
        )

        # 这些函数应该可以正常工作
        assert xlsx_validate_op("linear")
        assert "linear" in xlsx_list_ops()
        assert "inputs" in xlsx_get_op_info("linear")


class TestCppGoldenIntegration:
    """测试 C++ Golden 与注册表的集成"""

    def test_get_cpp_golden_ops(self):
        """获取所有标记 has_cpp_golden 的算子"""
        from aidevtools.ops.registry import get_cpp_golden_ops

        cpp_ops = get_cpp_golden_ops()
        assert "matmul" in cpp_ops
        assert "softmax" in cpp_ops
        assert "layernorm" in cpp_ops
        assert "transpose" in cpp_ops
        # 激活函数和逐元素运算现在也有 C++ golden
        assert "relu" in cpp_ops
        assert "gelu" in cpp_ops
        assert "sigmoid" in cpp_ops
        assert "tanh" in cpp_ops
        assert "silu" in cpp_ops
        assert "add" in cpp_ops
        assert "mul" in cpp_ops
        assert "div" in cpp_ops
        # batchnorm 没有 C++ golden
        assert "batchnorm" not in cpp_ops

    def test_check_cpp_golden_registered(self):
        """检查 C++ golden 注册状态 (检查类是否有 cpu_golden 方法)"""
        from aidevtools.ops.registry import check_cpp_golden_registered

        # 检查状态 (现在检查类是否有 cpu_golden 方法)
        status = check_cpp_golden_registered()
        assert status["matmul"] is True
        assert status["softmax"] is True
        assert status["layernorm"] is True
        assert status["transpose"] is True
