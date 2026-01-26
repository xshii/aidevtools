"""测试 torch_backend 模块"""

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from aidevtools.torch_backend import TorchGoldenBackend, TorchBackendConfig, golden_mode


class TestGoldenMode:
    """golden_mode context manager 测试"""

    def test_basic_linear(self):
        """基础 linear 劫持"""
        from aidevtools.ops.cpu_golden import is_cpu_golden_available

        if not is_cpu_golden_available():
            pytest.skip("CPU golden not available")

        x = torch.randn(2, 64)
        w = torch.randn(128, 64)
        b = torch.randn(128)

        # 原始 torch
        y_torch = F.linear(x, w, b)

        # 使用 golden
        with golden_mode(golden="python") as backend:
            y_golden = F.linear(x, w, b)

        # 形状应该相同
        assert y_golden.shape == y_torch.shape
        # cpu_golden 使用量化格式，放宽精度要求
        np.testing.assert_allclose(
            y_golden.numpy(), y_torch.numpy(), rtol=0.05, atol=0.3
        )

    def test_relu(self):
        """ReLU 劫持"""
        x = torch.randn(2, 64)

        with golden_mode(golden="python") as backend:
            y = F.relu(x)

        assert y.shape == x.shape
        assert (y >= 0).all()

    def test_compare_mode_fuzzy(self):
        """模糊比对"""
        x = torch.randn(2, 64)

        with golden_mode(golden="python", compare="fuzzy") as backend:
            y = F.relu(x)

        results = backend.get_compare_results()
        assert len(results) == 1
        assert results[0].status == "PASS"
        assert results[0].cosine_sim > 0.999

    def test_profile_enabled(self):
        """Paper Analysis profile"""
        x = torch.randn(2, 64)
        w = torch.randn(128, 64)

        with golden_mode(golden="python", profile=True) as backend:
            y = F.linear(x, w)
            y = F.relu(y)

        profiles = backend.get_profiles()
        assert len(profiles) == 2
        assert profiles[0].op_type == "linear"
        assert profiles[1].op_type == "relu"
        assert profiles[0].flops > 0


class TestQuantization:
    """量化测试"""

    def test_quantize_gfp16(self):
        """gfp16 量化"""
        x = torch.randn(2, 64)

        with golden_mode(golden="python", quantize="gfp16", compare="quantized") as backend:
            y = F.relu(x)

        results = backend.get_compare_results()
        assert len(results) == 1
        # 量化后 QSNR 应该较高
        assert results[0].qsnr_db > 10


class TestTorchBackendConfig:
    """配置测试"""

    def test_default_config(self):
        """默认配置"""
        config = TorchBackendConfig()
        assert config.golden_mode == "python"
        assert config.compare_mode == "none"
        assert config.quantize_type is None
        assert config.profile_enabled is False

    def test_configure(self):
        """动态配置"""
        backend = TorchGoldenBackend()
        backend.configure(
            golden_mode="python",
            compare_mode="fuzzy",
            profile_enabled=True,
        )
        assert backend.config.golden_mode == "python"
        assert backend.config.compare_mode == "fuzzy"
        assert backend.config.profile_enabled is True


class TestModuleIntegration:
    """nn.Module 集成测试"""

    def test_simple_model(self):
        """简单模型"""
        model = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )

        x = torch.randn(2, 64)

        with golden_mode(golden="python", compare="fuzzy", profile=True) as backend:
            y = model(x)

        assert y.shape == (2, 10)

        # 应该有 4 个算子记录 (linear, relu, linear)
        # 注意: nn.Linear 包含 linear 操作
        records = backend.get_records()
        assert len(records) >= 3

        # 比对应该全部通过
        results = backend.get_compare_results()
        assert all(r.status == "PASS" for r in results)

    def test_transformer_layer(self):
        """Transformer layer"""
        d_model = 64
        nhead = 4

        # 简化的 attention
        x = torch.randn(2, 8, d_model)  # [batch, seq, hidden]
        q = torch.randn(2, 8, d_model)
        k = torch.randn(2, 8, d_model)
        v = torch.randn(2, 8, d_model)

        with golden_mode(golden="python", profile=True) as backend:
            # Linear projections
            q_proj = F.linear(q, torch.randn(d_model, d_model))
            k_proj = F.linear(k, torch.randn(d_model, d_model))
            v_proj = F.linear(v, torch.randn(d_model, d_model))

            # Softmax (simplified)
            attn = F.softmax(q_proj @ k_proj.transpose(-2, -1), dim=-1)

            # Output
            out = F.linear(x, torch.randn(d_model, d_model))
            out = F.gelu(out)

        profiles = backend.get_profiles()
        # 应该有: 3x linear (q,k,v) + softmax + linear + gelu = 6
        assert len(profiles) >= 5
