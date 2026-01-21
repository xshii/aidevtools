"""全局常量定义

集中管理项目中的魔法数字和默认值。
"""

# ============================================================
# BFP (Block Floating Point) 配置
# ============================================================

# BFP16: 16元素块，8位尾数
BFP16_BLOCK_SIZE = 16
BFP16_MANTISSA_BITS = 8

# BFP8: 32元素块，4位尾数
BFP8_BLOCK_SIZE = 32
BFP8_MANTISSA_BITS = 4

# BFP4: 64元素块，2位尾数
BFP4_BLOCK_SIZE = 64
BFP4_MANTISSA_BITS = 2

# 默认配置
DEFAULT_BFP_BLOCK_SIZE = BFP16_BLOCK_SIZE
DEFAULT_BFP_MANTISSA_BITS = BFP16_MANTISSA_BITS


# ============================================================
# 算子默认参数
# ============================================================

# LayerNorm / BatchNorm
DEFAULT_NORM_EPS = 1e-5

# Softmax
DEFAULT_SOFTMAX_AXIS = -1

# Attention
DEFAULT_ATTENTION_SCALE = None  # 自动计算为 1/sqrt(d_k)


# ============================================================
# 量化配置
# ============================================================

# GFloat 类型
GFLOAT_TYPES = ("gfp4", "gfp8", "gfp16")
DEFAULT_GFLOAT_TYPE = "gfp16"

# BFP 类型
BFP_TYPES = ("bfp4", "bfp8", "bfp16")
DEFAULT_BFP_TYPE = "bfp8"


# ============================================================
# 比对阈值
# ============================================================

# 模糊比对默认值
DEFAULT_ATOL = 1e-5
DEFAULT_RTOL = 1e-3
DEFAULT_MIN_QSNR = 30.0  # dB
DEFAULT_MIN_COSINE = 0.999
