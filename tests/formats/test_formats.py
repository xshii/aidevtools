"""格式模块测试"""
import pytest
import numpy as np
from pathlib import Path

from aidevtools.formats.base import load, save

class TestRawFormat:
    """Raw 格式测试"""

    def test_save_load(self, tmp_workspace, sample_data):
        """保存和加载"""
        path = str(tmp_workspace / "test.bin")
        save(path, sample_data, format="raw")
        loaded = load(path, format="raw", dtype=np.float32, shape=sample_data.shape)
        assert np.allclose(sample_data, loaded)

    def test_load_reshape(self, tmp_workspace):
        """加载时 reshape"""
        data = np.arange(24, dtype=np.float32)
        path = str(tmp_workspace / "test.bin")
        save(path, data, format="raw")
        loaded = load(path, format="raw", dtype=np.float32, shape=(2, 3, 4))
        assert loaded.shape == (2, 3, 4)

class TestNumpyFormat:
    """Numpy 格式测试"""

    def test_npy(self, tmp_workspace, sample_data):
        """npy 格式"""
        path = str(tmp_workspace / "test.npy")
        save(path, sample_data, format="numpy")
        loaded = load(path, format="numpy")
        assert np.allclose(sample_data, loaded)

    def test_npz(self, tmp_workspace, sample_data):
        """npz 格式"""
        path = str(tmp_workspace / "test.npz")
        save(path, sample_data, format="numpy")
        loaded = load(path, format="numpy")
        assert np.allclose(sample_data, loaded)
