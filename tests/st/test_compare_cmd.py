"""Compare 命令系统测试

通过命令行接口进行端到端测试
"""
import pytest
import csv
import numpy as np
from pathlib import Path


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


class TestCompareCsvCmd:
    """csv 子命令测试"""

    def setup_method(self):
        from aidevtools.trace.tracer import clear
        clear()

    def test_gen_csv(self, tmp_path):
        """生成 CSV"""
        from aidevtools.commands.compare import cmd_compare
        from aidevtools.trace.tracer import trace, _records

        # 先记录一些操作
        @trace
        def test_op(x):
            return x * 2

        x = np.random.randn(2, 4).astype(np.float32)
        test_op(x)

        ret = cmd_compare(
            action="csv",
            output=str(tmp_path),
            model="test",
        )

        assert ret == 0
        csv_files = list(tmp_path.glob("*_compare.csv"))
        assert len(csv_files) == 1


class TestCompareClearCmd:
    """clear 子命令测试"""

    def test_clear(self):
        """清空记录"""
        from aidevtools.commands.compare import cmd_compare
        from aidevtools.trace.tracer import trace, _records, clear

        clear()

        @trace
        def test_op(x):
            return x

        test_op(np.array([1, 2, 3]))
        assert len(_records) == 1

        ret = cmd_compare(action="clear")
        assert ret == 0
        assert len(_records) == 0


class TestCompareRunCmd:
    """run 子命令测试"""

    def test_run_without_csv(self):
        """缺少 CSV 参数"""
        from aidevtools.commands.compare import cmd_compare

        ret = cmd_compare(action="run", csv="")
        assert ret == 1

    def test_run_with_csv(self, tmp_path):
        """运行比数"""
        from aidevtools.commands.compare import cmd_compare

        # 创建测试数据
        golden = np.random.randn(2, 4).astype(np.float32)
        golden_path = tmp_path / "op_golden.bin"
        result_path = tmp_path / "op_result.bin"
        golden.tofile(golden_path)
        golden.tofile(result_path)

        # 创建 CSV
        csv_path = tmp_path / "compare.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "op_name", "mode", "input_bin", "weight_bin", "golden_bin",
                "result_bin", "dtype", "shape", "qtype", "skip", "note"
            ])
            writer.writeheader()
            writer.writerow({
                "op_name": "test_op",
                "mode": "single",
                "input_bin": "",
                "weight_bin": "",
                "golden_bin": str(golden_path),
                "result_bin": str(result_path),
                "dtype": "float32",
                "shape": "2,4",
                "qtype": "",
                "skip": "false",
                "note": "",
            })

        ret = cmd_compare(action="run", csv=str(csv_path))
        assert ret == 0

        # 检查结果
        result_csv = csv_path.with_name("compare_result.csv")
        assert result_csv.exists()


class TestCompareArchiveCmd:
    """archive 子命令测试"""

    def test_archive_without_csv(self):
        """缺少 CSV 参数"""
        from aidevtools.commands.compare import cmd_compare

        ret = cmd_compare(action="archive", csv="")
        assert ret == 1

    def test_archive(self, tmp_path):
        """打包归档"""
        from aidevtools.commands.compare import cmd_compare

        csv_path = tmp_path / "compare.csv"
        csv_path.write_text("op_name\ntest\n")

        ret = cmd_compare(action="archive", csv=str(csv_path))
        assert ret == 0

        zip_path = csv_path.with_suffix(".zip")
        assert zip_path.exists()


class TestCompareUnknownCmd:
    """未知子命令测试"""

    def test_unknown_action(self):
        """未知操作"""
        from aidevtools.commands.compare import cmd_compare

        ret = cmd_compare(action="unknown_action")
        assert ret == 1
