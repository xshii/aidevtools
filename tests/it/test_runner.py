"""Runner 集成测试"""
import pytest
import csv
import numpy as np
from pathlib import Path


class TestRunCompare:
    """run_compare 集成测试"""

    def setup_method(self):
        """每个测试前清理"""
        from aidevtools.ops.base import clear
        clear()

    def _create_test_data(self, tmp_path, add_noise=False):
        """创建测试数据和 CSV"""
        # 创建 golden 和 result 数据
        golden = np.random.randn(2, 8, 64).astype(np.float32)
        if add_noise:
            result = golden + np.random.randn(*golden.shape).astype(np.float32) * 1e-6
        else:
            result = golden.copy()

        # 保存文件
        golden_path = tmp_path / "test_op_golden.bin"
        result_path = tmp_path / "test_op_result.bin"
        golden.tofile(golden_path)
        result.tofile(result_path)

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
                "shape": "2,8,64",
                "qtype": "",
                "skip": "false",
                "note": "",
            })

        return csv_path, golden, result

    def test_run_compare_pass(self, tmp_path):
        """比对通过"""
        from aidevtools.tools.compare.runner import run_compare

        csv_path, _, _ = self._create_test_data(tmp_path, add_noise=False)
        run_compare(str(csv_path), output_dir=str(tmp_path / "details"))

        # 检查结果文件
        result_csv = csv_path.with_name("compare_result.csv")
        assert result_csv.exists()

        with open(result_csv) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            assert len(rows) == 1
            assert rows[0]["status"] == "PASS"

    def test_run_compare_with_noise(self, tmp_path):
        """带噪声比对"""
        from aidevtools.tools.compare.runner import run_compare

        csv_path, _, _ = self._create_test_data(tmp_path, add_noise=True)
        run_compare(str(csv_path), output_dir=str(tmp_path / "details"))

        result_csv = csv_path.with_name("compare_result.csv")
        with open(result_csv) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            # 微小噪声应该通过
            assert rows[0]["status"] == "PASS"
            # QSNR 应该很高
            qsnr = float(rows[0]["qsnr"])
            assert qsnr > 40

    def test_run_compare_fail(self, tmp_path):
        """比对失败"""
        from aidevtools.tools.compare.runner import run_compare

        # 创建差异较大的数据
        golden = np.random.randn(2, 8, 64).astype(np.float32)
        result = golden + 0.1  # 较大误差

        golden_path = tmp_path / "test_op_golden.bin"
        result_path = tmp_path / "test_op_result.bin"
        golden.tofile(golden_path)
        result.tofile(result_path)

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
                "shape": "2,8,64",
                "qtype": "",
                "skip": "false",
                "note": "",
            })

        run_compare(str(csv_path), output_dir=str(tmp_path / "details"))

        result_csv = csv_path.with_name("compare_result.csv")
        with open(result_csv) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            assert rows[0]["status"] == "FAIL"

    def test_run_compare_skip(self, tmp_path):
        """跳过标记的用例"""
        from aidevtools.tools.compare.runner import run_compare

        golden = np.random.randn(2, 8, 64).astype(np.float32)
        golden_path = tmp_path / "test_op_golden.bin"
        result_path = tmp_path / "test_op_result.bin"
        golden.tofile(golden_path)
        golden.tofile(result_path)

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
                "shape": "2,8,64",
                "qtype": "",
                "skip": "true",  # 跳过
                "note": "",
            })

        run_compare(str(csv_path), output_dir=str(tmp_path / "details"))

        # 跳过的用例不会生成结果
        result_csv = csv_path.with_name("compare_result.csv")
        assert not result_csv.exists()

    def test_run_compare_op_filter(self, tmp_path):
        """算子过滤"""
        from aidevtools.tools.compare.runner import run_compare

        golden = np.random.randn(2, 4).astype(np.float32)

        # 创建两个算子的数据
        for op in ["op1", "op2"]:
            (tmp_path / f"{op}_golden.bin").write_bytes(golden.tobytes())
            (tmp_path / f"{op}_result.bin").write_bytes(golden.tobytes())

        csv_path = tmp_path / "compare.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "op_name", "mode", "input_bin", "weight_bin", "golden_bin",
                "result_bin", "dtype", "shape", "qtype", "skip", "note"
            ])
            writer.writeheader()
            for op in ["op1", "op2"]:
                writer.writerow({
                    "op_name": op,
                    "mode": "single",
                    "input_bin": "",
                    "weight_bin": "",
                    "golden_bin": str(tmp_path / f"{op}_golden.bin"),
                    "result_bin": str(tmp_path / f"{op}_result.bin"),
                    "dtype": "float32",
                    "shape": "2,4",
                    "qtype": "",
                    "skip": "false",
                    "note": "",
                })

        # 只跑 op1
        run_compare(str(csv_path), output_dir=str(tmp_path / "details"), op_filter="op1")

        result_csv = csv_path.with_name("compare_result.csv")
        with open(result_csv) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            assert len(rows) == 1
            assert rows[0]["op_name"] == "op1"


class TestArchive:
    """archive 打包测试"""

    def test_archive(self, tmp_path):
        """打包归档"""
        from aidevtools.tools.compare.runner import archive

        # 创建测试文件
        csv_path = tmp_path / "compare.csv"
        csv_path.write_text("op_name,status\ntest,PASS\n")

        details_dir = tmp_path / "details"
        details_dir.mkdir()
        (details_dir / "test.txt").write_text("detail content")

        # 打包
        zip_path = archive(str(csv_path))

        assert Path(zip_path).exists()
        assert zip_path.endswith(".zip")

        # 验证内容
        import zipfile
        with zipfile.ZipFile(zip_path) as zf:
            names = zf.namelist()
            assert "compare.csv" in names
            assert "details/test.txt" in names
