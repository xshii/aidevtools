"""Export 集成测试"""
import json
import numpy as np


class TestExportFailedCases:
    """export_failed_cases 测试"""

    def test_export_failed_blocks(self, tmp_path):
        """导出低 QSNR 片段"""
        from aidevtools.tools.compare.export import ExportConfig, export_failed_cases

        # 创建测试数据
        golden = np.arange(1024, dtype=np.float32)

        # 模拟 blocks 结果
        blocks = [
            {"offset": 0, "size": 1024, "qsnr": 50.0, "max_abs": 1e-6, "passed": True},
            {"offset": 1024, "size": 1024, "qsnr": 10.0, "max_abs": 10.0, "passed": False},  # 低 QSNR
            {"offset": 2048, "size": 1024, "qsnr": 45.0, "max_abs": 1e-5, "passed": True},
        ]

        config = ExportConfig(
            output_dir=str(tmp_path),
            op_name="test_op",
            qsnr_threshold=20.0
        )
        exported = export_failed_cases(golden, blocks, config)

        assert exported == 1  # 只导出 1 个低于阈值的

        # 检查导出文件
        case_dir = tmp_path / "test_op" / "failed_cases"
        assert case_dir.exists()

        # 应该有 bin 和 json 文件
        bin_files = list(case_dir.glob("*.bin"))
        json_files = list(case_dir.glob("*.json"))
        assert len(bin_files) == 1
        assert len(json_files) == 1

        # 验证 json 内容
        with open(json_files[0], encoding="utf-8") as f:
            param = json.load(f)
            assert param["op_name"] == "test_op"
            assert param["qsnr"] == 10.0

    def test_export_no_failed(self, tmp_path):
        """无失败用例时不导出"""
        from aidevtools.tools.compare.export import ExportConfig, export_failed_cases

        golden = np.arange(100, dtype=np.float32)

        blocks = [
            {"offset": 0, "size": 400, "qsnr": 50.0, "max_abs": 0, "passed": True},
        ]

        config = ExportConfig(
            output_dir=str(tmp_path),
            op_name="test_op",
            qsnr_threshold=20.0
        )
        exported = export_failed_cases(golden, blocks, config)

        assert exported == 0

    def test_export_multiple_failed(self, tmp_path):
        """多个失败片段"""
        from aidevtools.tools.compare.export import ExportConfig, export_failed_cases

        golden = np.arange(1000, dtype=np.float32)

        blocks = [
            {"offset": i * 100, "size": 100, "qsnr": 5.0, "max_abs": 5.0, "passed": False}
            for i in range(10)
        ]

        config = ExportConfig(
            output_dir=str(tmp_path),
            op_name="test_op",
            qsnr_threshold=20.0
        )
        exported = export_failed_cases(golden, blocks, config)

        assert exported == 10

        case_dir = tmp_path / "test_op" / "failed_cases"
        bin_files = list(case_dir.glob("*.bin"))
        assert len(bin_files) == 10
