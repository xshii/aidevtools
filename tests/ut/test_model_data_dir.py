"""Model data_dir 自动加载测试"""
import numpy as np
import pytest

from aidevtools.datagen import Model
from aidevtools.frontend.types import PrecisionConfig


PRECISION = PrecisionConfig(
    input_dtype="fp16",
    weight_dtype="bfp4",
    compute_dtype="fp32",
    output_dtype="bfp8",
    qa_aware=True,
    qa_center=1.0,
    qa_amplitude=0.5,
)


class TestModelDataDir:
    """Model data_dir 自动加载测试"""

    def test_export_then_reload(self, tmp_workspace):
        """export → data_dir reload 两次应产生相同输出"""
        # 生成 + export
        with Model(seed=42, precision=PRECISION, qtype="bfp8") as m1:
            x = m1.input((1, 4, 16))
            y = m1.linear(x, out_features=16)
        m1.export(str(tmp_workspace), bm="test")

        # 从 data_dir 加载 (第一次)
        with Model(data_dir=str(tmp_workspace), bm="test",
                    qtype="bfp8", precision=PRECISION) as m2:
            x = m2.input((1, 4, 16))
            y = m2.linear(x, out_features=16)
        output1 = m2.final_output.copy()

        # 从 data_dir 加载 (第二次) — 应完全一致
        with Model(data_dir=str(tmp_workspace), bm="test",
                    qtype="bfp8", precision=PRECISION) as m3:
            x = m3.input((1, 4, 16))
            y = m3.linear(x, out_features=16)
        output2 = m3.final_output

        assert output1.shape == (1, 4, 16)
        assert np.allclose(output1, output2)

    def test_data_dir_loads_input(self, tmp_workspace):
        """data_dir 模式应从文件加载 input"""
        with Model(seed=42, precision=PRECISION, qtype="bfp8") as m1:
            x = m1.input((1, 4, 16))
        m1.export(str(tmp_workspace), bm="enc")
        orig_input = m1.outputs  # empty since no op, but we can check tensors
        orig_input_arr = list(m1.tensors.values())[0].array.copy()

        with Model(data_dir=str(tmp_workspace), bm="enc",
                    qtype="bfp8", precision=PRECISION) as m2:
            x = m2.input((1, 4, 16))

        # Loaded input should approximate the original (quantization loss)
        assert x.golden is not None
        assert x.golden.shape == (1, 4, 16)

    def test_data_dir_loads_weights(self, tmp_workspace):
        """data_dir 模式应从文件加载 weight"""
        with Model(seed=42, precision=PRECISION, qtype="bfp8") as m1:
            x = m1.input((1, 4, 16))
            y = m1.linear(x, out_features=16)
        m1.export(str(tmp_workspace), bm="enc")

        # 收集原始 weight 名
        weight_names = [n for n in m1.tensors if "weight" in n]
        assert len(weight_names) > 0

        with Model(data_dir=str(tmp_workspace), bm="enc",
                    qtype="bfp8", precision=PRECISION) as m2:
            x = m2.input((1, 4, 16))
            y = m2.linear(x, out_features=16)

        # 验证 data_dir 中有文件被加载
        assert m2._loaded is not None
        assert len(m2._loaded) > 0

    def test_data_dir_corrupted_produces_diff(self, tmp_workspace):
        """篡改 data_dir 中的文件应导致不同输出"""
        import shutil

        golden_dir = tmp_workspace / "golden"
        corrupt_dir = tmp_workspace / "corrupt"

        # 生成 + export
        with Model(seed=42, precision=PRECISION, qtype="bfp8") as m1:
            x = m1.input((1, 4, 16))
            y = m1.linear(x, out_features=16)
        m1.export(str(golden_dir), bm="enc")
        orig_output = m1.final_output.copy()

        # 复制并篡改
        shutil.copytree(golden_dir, corrupt_dir)
        for f in corrupt_dir.glob("*weight*.bin"):
            raw = bytearray(f.read_bytes())
            for i in range(min(16, len(raw))):
                raw[i] ^= 0xFF
            f.write_bytes(bytes(raw))

        # 从篡改目录加载
        with Model(data_dir=str(corrupt_dir), bm="enc",
                    qtype="bfp8", precision=PRECISION) as m2:
            x = m2.input((1, 4, 16))
            y = m2.linear(x, out_features=16)
        corrupt_output = m2.final_output

        # 输出应不同
        assert not np.allclose(orig_output, corrupt_output, atol=1e-3)

    def test_data_dir_bm_filter(self, tmp_workspace):
        """data_dir 只加载匹配 bm 前缀的文件"""
        with Model(seed=42, precision=PRECISION, qtype="bfp8") as m1:
            x = m1.input((1, 4, 16))
            y = m1.linear(x, out_features=16)
        m1.export(str(tmp_workspace), bm="enc")

        # 用不同的 bm 加载，应找不到任何文件
        with Model(data_dir=str(tmp_workspace), bm="dec",
                    qtype="bfp8", precision=PRECISION) as m2:
            pass  # noqa
        assert m2._loaded is not None
        assert len(m2._loaded) == 0

    def test_data_dir_nonexistent_raises(self):
        """不存在的 data_dir 应报错"""
        with pytest.raises(ValueError, match="不是目录"):
            with Model(data_dir="/nonexistent/path", bm="enc",
                        qtype="bfp8", precision=PRECISION) as m:
                pass

    def test_model_without_data_dir(self, tmp_workspace):
        """不设 data_dir 时应正常生成随机数"""
        with Model(seed=42, precision=PRECISION, qtype="bfp8") as m:
            x = m.input((1, 4, 16))
            y = m.linear(x, out_features=16)

        assert m._loaded is None
        assert m.final_output is not None
        assert m.final_output.shape == (1, 4, 16)

    def test_multi_op_replay(self, tmp_workspace):
        """多算子 replay: 两次 data_dir 加载输出应完全一致"""
        # 生成 + export
        with Model(seed=42, precision=PRECISION, qtype="bfp8") as m1:
            x = m1.input((1, 4, 16))
            y = m1.linear(x, out_features=32)
            y = m1.gelu(y)
            y = m1.linear(y, out_features=16)
        m1.export(str(tmp_workspace), bm="enc")

        # 回放 1
        with Model(data_dir=str(tmp_workspace), bm="enc",
                    qtype="bfp8", precision=PRECISION) as m2:
            x = m2.input((1, 4, 16))
            y = m2.linear(x, out_features=32)
            y = m2.gelu(y)
            y = m2.linear(y, out_features=16)
        outputs1 = [o.golden.copy() for o in m2.outputs]

        # 回放 2
        with Model(data_dir=str(tmp_workspace), bm="enc",
                    qtype="bfp8", precision=PRECISION) as m3:
            x = m3.input((1, 4, 16))
            y = m3.linear(x, out_features=32)
            y = m3.gelu(y)
            y = m3.linear(y, out_features=16)
        outputs2 = [o.golden for o in m3.outputs]

        assert len(outputs1) == len(outputs2) == 3  # linear, gelu, linear
        for i, (o1, o2) in enumerate(zip(outputs1, outputs2)):
            assert np.allclose(o1, o2), \
                f"Op {i}: max diff = {np.max(np.abs(o1 - o2))}"
