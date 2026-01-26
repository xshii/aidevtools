"""Compare 命令系统测试

通过命令行接口进行端到端测试
"""
import numpy as np
import pytest


class TestCompareSingleCmd:
    """single 子命令测试"""

    def test_single_pass(self, tmp_path):
        """单次比对 - 通过"""
        from aidevtools.commands.compare import cmd_compare

        # 创建相同的测试数据
        data = np.random.randn(2, 8, 64).astype(np.float32)
        golden_path = tmp_path / "golden.bin"
        result_path = tmp_path / "result.bin"
        data.tofile(golden_path)
        data.tofile(result_path)

        ret = cmd_compare(
            action="single",
            golden=str(golden_path),
            result=str(result_path),
            dtype="float32",
            shape="2,8,64",
        )

        assert ret == 0  # 通过

    def test_single_fail(self, tmp_path):
        """单次比对 - 失败"""
        from aidevtools.commands.compare import cmd_compare

        golden = np.random.randn(2, 8, 64).astype(np.float32)
        result = golden + 0.1  # 较大差异

        golden_path = tmp_path / "golden.bin"
        result_path = tmp_path / "result.bin"
        golden.tofile(golden_path)
        result.tofile(result_path)

        ret = cmd_compare(
            action="single",
            golden=str(golden_path),
            result=str(result_path),
            dtype="float32",
            shape="2,8,64",
        )

        assert ret == 1  # 失败

    def test_single_missing_args(self):
        """缺少参数"""
        from aidevtools.commands.compare import cmd_compare

        ret = cmd_compare(action="single", golden="", result="")
        assert ret == 1


class TestCompareFuzzyCmd:
    """fuzzy 子命令测试"""

    def test_fuzzy_compare(self, tmp_path):
        """模糊比对"""
        from aidevtools.commands.compare import cmd_compare

        golden = np.random.randn(100).astype(np.float32)
        result = golden + np.random.randn(100).astype(np.float32) * 0.01

        golden_path = tmp_path / "golden.bin"
        result_path = tmp_path / "result.bin"
        golden.tofile(golden_path)
        result.tofile(result_path)

        ret = cmd_compare(
            action="fuzzy",
            golden=str(golden_path),
            result=str(result_path),
            dtype="float32",
        )

        assert ret == 0


class TestCompareConvertCmd:
    """convert 子命令测试"""

    def test_convert_to_float16(self, tmp_path):
        """转换为 float16"""
        from aidevtools.commands.compare import cmd_compare

        data = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        input_path = tmp_path / "input.bin"
        output_path = tmp_path / "output.bin"
        data.tofile(input_path)

        ret = cmd_compare(
            action="convert",
            golden=str(input_path),
            output=str(output_path),
            target_dtype="float16",
            dtype="float32",
        )

        assert ret == 0
        assert output_path.exists()

        # 验证输出是 float16
        converted = np.fromfile(output_path, dtype=np.float16)
        assert len(converted) == 3

    def test_convert_unknown_dtype(self, tmp_path):
        """未知目标类型"""
        from aidevtools.commands.compare import cmd_compare

        data = np.array([1.0], dtype=np.float32)
        input_path = tmp_path / "input.bin"
        data.tofile(input_path)

        ret = cmd_compare(
            action="convert",
            golden=str(input_path),
            target_dtype="unknown_type",
            dtype="float32",
        )

        assert ret == 1


class TestCompareQtypesCmd:
    """qtypes 子命令测试"""

    def test_list_qtypes(self, capsys):
        """列出量化类型"""
        from aidevtools.commands.compare import cmd_compare

        ret = cmd_compare(action="qtypes")
        assert ret == 0

        captured = capsys.readouterr()
        assert "float16" in captured.out
        assert "gfloat16" in captured.out


class TestCompareClearCmd:
    """clear 子命令测试"""

    def test_clear(self):
        """清空记录"""
        from aidevtools.ops import _functional as F
        from aidevtools.commands.compare import cmd_compare
        from aidevtools.ops.base import clear, get_records
        from aidevtools.ops.cpu_golden import is_cpu_golden_available

        if not is_cpu_golden_available():
            pytest.skip("CPU golden not available")

        clear()

        # 使用 F 触发记录 (使用有 cpu_golden 的算子)
        x = np.random.randn(2, 4).astype(np.float32)
        F.softmax(x)
        assert len(get_records()) == 1

        ret = cmd_compare(action="clear")
        assert ret == 0
        assert len(get_records()) == 0


class TestCompareUnknownCmd:
    """未知子命令测试"""

    def test_unknown_action(self):
        """未知操作"""
        from aidevtools.commands.compare import cmd_compare

        ret = cmd_compare(action="unknown_action")
        assert ret == 1


class TestCompareDumpCmd:
    """dump 子命令测试"""

    def setup_method(self):
        from aidevtools.ops.base import clear
        clear()

    def test_dump(self, tmp_path):
        """导出数据"""
        from aidevtools.ops import _functional as F
        from aidevtools.commands.compare import cmd_compare
        from aidevtools.ops.cpu_golden import is_cpu_golden_available

        if not is_cpu_golden_available():
            pytest.skip("CPU golden not available")

        x = np.random.randn(2, 4).astype(np.float32)
        F.softmax(x)

        ret = cmd_compare(action="dump", output=str(tmp_path))
        assert ret == 0

        # 检查文件生成
        assert (tmp_path / "softmax_0_golden.bin").exists()
