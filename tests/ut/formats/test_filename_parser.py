"""文件名解读器单元测试"""
from aidevtools.formats.filename_parser import parse_filename, infer_fmt


class TestParseFilename:
    """parse_filename 测试"""

    def test_basic_bfpp8(self):
        r = parse_filename("softmax_bfpp8_2x16x64.txt")
        assert r["op"] == "softmax"
        assert r["qtype"] == "bfpp8"
        assert r["shape"] == (2, 16, 64)
        assert r["is_result"] is False
        assert r["fmt"] == "hex_text"
        assert r["ext"] == ".txt"

    def test_result_suffix(self):
        r = parse_filename("softmax_bfpp8_2x16x64_result.txt")
        assert r["op"] == "softmax"
        assert r["qtype"] == "bfpp8"
        assert r["shape"] == (2, 16, 64)
        assert r["is_result"] is True

    def test_op_with_underscore(self):
        """算子名含下划线"""
        r = parse_filename("linear_0_gfloat8_64x64.txt")
        assert r["op"] == "linear_0"
        assert r["qtype"] == "gfloat8"
        assert r["shape"] == (64, 64)

    def test_op_with_multiple_underscores(self):
        """算子名含多段下划线"""
        r = parse_filename("batch_norm_2_bfpp16_128.txt")
        assert r["op"] == "batch_norm_2"
        assert r["qtype"] == "bfpp16"
        assert r["shape"] == (128,)

    def test_1d_shape(self):
        r = parse_filename("relu_bfpp4_256.txt")
        assert r["shape"] == (256,)

    def test_gfloat16(self):
        r = parse_filename("conv_gfloat16_3x3x64x128.txt")
        assert r["qtype"] == "gfloat16"
        assert r["shape"] == (3, 3, 64, 128)

    def test_float32(self):
        r = parse_filename("add_float32_1024.bin")
        assert r["qtype"] == "float32"
        assert r["fmt"] == "raw"
        assert r["ext"] == ".bin"

    def test_float16_result(self):
        r = parse_filename("matmul_float16_32x64_result.bin")
        assert r["op"] == "matmul"
        assert r["qtype"] == "float16"
        assert r["shape"] == (32, 64)
        assert r["is_result"] is True
        assert r["fmt"] == "raw"

    def test_no_match_returns_none(self):
        """不符合约定的文件名 → None"""
        assert parse_filename("random_file.txt") is None
        assert parse_filename("data.bin") is None
        assert parse_filename("no_qtype_here_64.txt") is None

    def test_full_path(self):
        """支持完整路径 (取 basename)"""
        r = parse_filename("/some/dir/softmax_bfpp8_64.txt")
        assert r["op"] == "softmax"
        assert r["qtype"] == "bfpp8"
        assert r["shape"] == (64,)

    def test_npy_extension(self):
        r = parse_filename("op_bfpp8_64.npy")
        assert r["fmt"] == "numpy"

    def test_unknown_extension(self):
        """未知扩展名 → fmt=raw"""
        r = parse_filename("op_bfpp8_64.dat")
        assert r["fmt"] == "raw"

    def test_bfp_placeholder(self):
        """预留 bfp 格式名 — 文件名解析能识别"""
        r = parse_filename("softmax_bfp8_64.txt")
        assert r is not None
        assert r["qtype"] == "bfp8"
        assert r["shape"] == (64,)


class TestInferFmt:
    """infer_fmt 测试"""

    def test_txt(self):
        assert infer_fmt("a.txt") == "hex_text"

    def test_bin(self):
        assert infer_fmt("a.bin") == "raw"

    def test_npy(self):
        assert infer_fmt("a.npy") == "numpy"

    def test_npz(self):
        assert infer_fmt("a.npz") == "numpy"

    def test_unknown(self):
        assert infer_fmt("a.dat") == "raw"
