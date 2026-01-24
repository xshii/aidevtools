"""模糊比对单元测试"""
import pytest
import numpy as np
from pathlib import Path


class TestFuzzyCase:
    """FuzzyCase 测试"""

    def test_create_case(self, tmp_path):
        """创建用例"""
        from aidevtools.tools.compare.fuzzy import create_fuzzy_case

        case = create_fuzzy_case("test_conv", str(tmp_path))
        assert case.name == "test_conv"
        assert (tmp_path / "test_conv").exists()

    def test_set_input_weight(self, tmp_path):
        """设置输入和权重"""
        from aidevtools.tools.compare.fuzzy import FuzzyCase

        case = FuzzyCase("test", str(tmp_path))
        x = np.random.randn(2, 3, 4).astype(np.float64)  # 故意用 float64
        w = np.random.randn(4, 5).astype(np.float64)

        case.set_input("x", x)
        case.set_weight("w", w)

        # 应该转为 float32
        assert case.inputs["x"].dtype == np.float32
        assert case.weights["w"].dtype == np.float32

    def test_compute_golden(self, tmp_path):
        """计算 Golden"""
        from aidevtools.tools.compare.fuzzy import FuzzyCase

        case = FuzzyCase("test", str(tmp_path))
        x = np.random.randn(2, 3, 4).astype(np.float32)
        w = np.random.randn(4, 5).astype(np.float32)

        case.set_input("x", x)
        case.set_weight("w", w)
        case.set_compute(lambda inputs, weights: np.matmul(inputs["x"], weights["w"]))

        golden = case.compute_golden()
        expected = np.matmul(x, w)
        assert np.allclose(golden, expected)

    def test_compute_golden_without_fn(self, tmp_path):
        """未设置计算函数时报错"""
        from aidevtools.tools.compare.fuzzy import FuzzyCase

        case = FuzzyCase("test", str(tmp_path))
        with pytest.raises(ValueError, match="请先设置计算函数"):
            case.compute_golden()

    def test_export(self, tmp_path):
        """导出量化数据"""
        from aidevtools.tools.compare.fuzzy import FuzzyCase

        case = FuzzyCase("test", str(tmp_path))
        x = np.random.randn(2, 3).astype(np.float32)
        w = np.random.randn(3, 4).astype(np.float32)

        case.set_input("x", x)
        case.set_weight("w", w)
        case.set_compute(lambda inputs, weights: np.matmul(inputs["x"], weights["w"]))
        case.compute_golden()

        paths = case.export(qtype="float16")

        assert "golden" in paths
        assert "input_x" in paths
        assert "weight_w" in paths

        # 检查文件存在
        assert Path(paths["golden"]).exists()
        assert Path(paths["input_x"]).exists()
        assert Path(paths["weight_w"]).exists()

        # golden 应该是 fp32
        golden_data = np.fromfile(paths["golden"], dtype=np.float32)
        assert golden_data.shape[0] == 2 * 4  # 2x4 矩阵

        # 输入应该是 fp16
        input_data = np.fromfile(paths["input_x"], dtype=np.float16)
        assert input_data.shape[0] == 2 * 3

    def test_export_info(self, tmp_path):
        """导出用例信息"""
        from aidevtools.tools.compare.fuzzy import FuzzyCase

        case = FuzzyCase("test", str(tmp_path))
        x = np.random.randn(2, 3).astype(np.float32)

        case.set_input("x", x)
        case.set_compute(lambda inputs, weights: inputs["x"] * 2)
        case.compute_golden()

        info = case.export_info()
        assert info["name"] == "test"
        assert info["inputs"]["x"]["shape"] == [2, 3]
        assert info["golden_shape"] == [2, 3]

    def test_export_with_int8_meta(self, tmp_path):
        """导出带 meta 信息的量化数据"""
        from aidevtools.tools.compare.fuzzy import FuzzyCase
        from aidevtools.formats.quantize import register_quantize

        # 注册一个带 meta 的量化类型
        @register_quantize("test_int8")
        def to_test_int8(data, **kwargs):
            scale = np.max(np.abs(data)) / 127
            q_data = np.round(data / scale).astype(np.int8)
            return q_data, {"scale": float(scale), "zero_point": 0}

        case = FuzzyCase("test_meta", str(tmp_path))
        x = np.random.randn(4, 4).astype(np.float32)
        w = np.random.randn(4, 4).astype(np.float32)

        case.set_input("x", x)
        case.set_weight("w", w)
        case.set_compute(lambda inputs, weights: np.matmul(inputs["x"], weights["w"]))
        case.compute_golden()

        case.export(qtype="test_int8")

        # 检查 meta 文件
        input_meta_path = tmp_path / "test_meta" / "input_x.meta"
        weight_meta_path = tmp_path / "test_meta" / "weight_w.meta"

        assert input_meta_path.exists()
        assert weight_meta_path.exists()

        # 检查 meta 内容
        meta_content = input_meta_path.read_text()
        assert "scale=" in meta_content
        assert "zero_point=" in meta_content
