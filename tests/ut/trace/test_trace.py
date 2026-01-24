"""Trace 模块测试"""
import numpy as np

from aidevtools.trace.tracer import trace, dump, clear, _records

class TestTrace:
    """Trace 装饰器测试"""

    def setup_method(self):
        clear()

    def test_trace_basic(self, sample_data):
        """基本插桩"""
        @trace
        def my_op(x):
            return x * 2

        result = my_op(sample_data)
        assert np.allclose(result, sample_data * 2)
        assert len(_records) == 1
        assert _records[0]["name"] == "my_op_0"

    def test_trace_multiple(self, sample_data):
        """多次调用"""
        @trace
        def my_op(x):
            return x + 1

        my_op(sample_data)
        my_op(sample_data)
        assert len(_records) == 2
        assert _records[0]["name"] == "my_op_0"
        assert _records[1]["name"] == "my_op_1"

    def test_trace_with_name(self, sample_data):
        """自定义名称"""
        @trace(name="conv2d")
        def my_conv(x, w):
            return x * w

        w = np.ones_like(sample_data)
        my_conv(sample_data, w)
        assert _records[0]["name"] == "conv2d_0"
        assert _records[0]["weight"] is not None

class TestDump:
    """数据导出测试"""

    def setup_method(self):
        clear()

    def test_dump(self, tmp_workspace, sample_data):
        """导出文件"""
        @trace
        def my_op(x):
            return x * 2

        my_op(sample_data)
        dump(str(tmp_workspace))

        assert (tmp_workspace / "my_op_0_golden.bin").exists()
        assert (tmp_workspace / "my_op_0_input.bin").exists()

