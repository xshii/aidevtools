"""格式模块测试"""
import pytest
import numpy as np

from aidevtools.formats.base import load, save

class TestRawFormat:
    """Raw 格式测试"""

    def test_save_load(self, tmp_workspace, sample_data):
        """保存和加载"""
        path = str(tmp_workspace / "test.bin")
        save(path, sample_data, fmt="raw")
        loaded = load(path, fmt="raw", dtype=np.float32, shape=sample_data.shape)
        assert np.allclose(sample_data, loaded)

    def test_load_reshape(self, tmp_workspace):
        """加载时 reshape"""
        data = np.arange(24, dtype=np.float32)
        path = str(tmp_workspace / "test.bin")
        save(path, data, fmt="raw")
        loaded = load(path, fmt="raw", dtype=np.float32, shape=(2, 3, 4))
        assert loaded.shape == (2, 3, 4)

class TestNumpyFormat:
    """Numpy 格式测试"""

    def test_npy(self, tmp_workspace, sample_data):
        """npy 格式"""
        path = str(tmp_workspace / "test.npy")
        save(path, sample_data, fmt="numpy")
        loaded = load(path, fmt="numpy")
        assert np.allclose(sample_data, loaded)

    def test_npz(self, tmp_workspace, sample_data):
        """npz 格式"""
        path = str(tmp_workspace / "test.npz")
        save(path, sample_data, fmt="numpy")
        loaded = load(path, fmt="numpy")
        assert np.allclose(sample_data, loaded)


class TestFormatEdgeCases:
    """格式边界测试"""

    def test_unknown_format_save(self, tmp_workspace, sample_data):
        """未知格式保存"""
        path = str(tmp_workspace / "test.bin")
        with pytest.raises(ValueError, match="未知格式"):
            save(path, sample_data, fmt="unknown_format")

    def test_unknown_format_load(self, tmp_workspace, sample_data):
        """未知格式加载"""
        path = str(tmp_workspace / "test.bin")
        save(path, sample_data, fmt="raw")
        with pytest.raises(ValueError, match="未知格式"):
            load(path, fmt="unknown_format")

    def test_auto_detect_npy(self, tmp_workspace, sample_data):
        """自动检测 npy 格式"""
        path = str(tmp_workspace / "test.npy")
        np.save(path, sample_data)
        # 使用 numpy 格式加载 .npy 文件
        loaded = load(path, fmt="numpy")
        assert np.allclose(sample_data, loaded)

    def test_raw_without_dtype(self, tmp_workspace, sample_data):
        """raw 格式未指定 dtype"""
        path = str(tmp_workspace / "test.bin")
        save(path, sample_data, fmt="raw")
        # 未指定 dtype 时默认为 float32
        loaded = load(path, fmt="raw", shape=sample_data.shape)
        assert loaded.dtype == np.float32
