"""pytest 配置"""
import pytest
import numpy as np

@pytest.fixture
def tmp_workspace(tmp_path):
    """临时工作目录"""
    return tmp_path

@pytest.fixture
def sample_data():
    """样例数据"""
    np.random.seed(42)
    return np.random.randn(1, 3, 8, 8).astype(np.float32)

@pytest.fixture
def golden_result(sample_data):
    """模拟 golden 结果"""
    return sample_data * 2

@pytest.fixture
def sim_result(golden_result):
    """模拟仿真器结果 (带少量误差)"""
    noise = np.random.randn(*golden_result.shape).astype(np.float32) * 1e-6
    return golden_result + noise
