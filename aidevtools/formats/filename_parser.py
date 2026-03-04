"""文件名解读器

从文件名中提取算子类型、量化类型、shape、是否为 result。

命名约定::

    {op}_{qtype}_{shape}.{ext}            — golden
    {op}_{qtype}_{shape}_result.{ext}     — result

示例::

    softmax_bfp8_2x16x64.txt              → op=softmax, qtype=bfp8, shape=(2,16,64), is_result=False
    linear_0_gfloat8_64x64_result.txt      → op=linear_0, qtype=gfloat8, shape=(64,64), is_result=True

正则分组::

    (?P<op>.+?)_(?P<qtype>QTYPES)_(?P<shape>\\d+(?:x\\d+)*)(?:_result)?(?P<ext>\\..+)$
"""
import os
import re
from typing import Dict, Optional, Tuple

# 已知量化类型 (长的在前，避免前缀匹配)
_KNOWN_QTYPES = [
    "gfloat16", "gfloat8", "gfloat4",
    "bfpp16", "bfpp8", "bfpp4",
    "bfp16", "bfp8", "bfp4",       # 预留: 用户通过 register_block_format 注册
    "float32", "float16",
]

_QTYPE_ALT = "|".join(_KNOWN_QTYPES)

# 主正则: 4 个命名分组 + 可选 _result
#   op     — 算子名 (允许含下划线, 如 linear_0, batch_norm)
#   qtype  — 量化类型
#   shape  — NxMxK 格式
#   ext    — 文件扩展名
_FILENAME_RE = re.compile(
    rf"^(?P<op>.+?)_(?P<qtype>{_QTYPE_ALT})_(?P<shape>\d+(?:x\d+)*)(?P<result>_result)?(?P<ext>\..+)$"
)

# 扩展名 → 格式名
_EXT_TO_FMT = {
    ".txt": "hex_text",
    ".bin": "raw",
    ".npy": "numpy",
    ".npz": "numpy",
}


def parse_filename(path: str) -> Optional[Dict]:
    """从文件名解读 op / qtype / shape / is_result / fmt

    Args:
        path: 文件路径 (取 basename 解析)

    Returns:
        解析成功返回 dict, 否则 None::

            {
                "op": "softmax",
                "qtype": "bfp8",
                "shape": (2, 16, 64),
                "is_result": False,
                "fmt": "hex_text",
                "ext": ".txt",
            }
    """
    basename = os.path.basename(path)
    m = _FILENAME_RE.match(basename)
    if not m:
        return None
    shape = tuple(int(d) for d in m.group("shape").split("x"))
    ext = m.group("ext")
    return {
        "op": m.group("op"),
        "qtype": m.group("qtype"),
        "shape": shape,
        "is_result": m.group("result") is not None,
        "fmt": _EXT_TO_FMT.get(ext, "raw"),
        "ext": ext,
    }


def infer_fmt(path: str) -> str:
    """从扩展名推断格式"""
    _, ext = os.path.splitext(path)
    return _EXT_TO_FMT.get(ext, "raw")
