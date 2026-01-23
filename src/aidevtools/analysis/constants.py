"""分析模块常量定义"""

# === 单位转换 ===
TFLOPS_TO_FLOPS = 1e12  # TFLOPS -> FLOPS
GFLOPS_TO_FLOPS = 1e9   # GFLOPS -> FLOPS
GBPS_TO_BPS = 1e9       # GB/s -> Bytes/s
US_TO_S = 1e-6          # 微秒 -> 秒
S_TO_US = 1e6           # 秒 -> 微秒
MB_TO_BYTES = 1024 * 1024
GB_TO_BYTES = 1024 * 1024 * 1024

# === 计算单元 ===
UNIT_CUBE = "cube"
UNIT_VECTOR = "vector"
UNIT_DMA = "dma"

# === 访存模式 ===
PATTERN_SEQUENTIAL = "sequential"
PATTERN_STRIDED = "strided"
PATTERN_RANDOM = "random"

# === 瓶颈类型 ===
BOTTLENECK_COMPUTE = "compute"
BOTTLENECK_MEMORY = "memory"
