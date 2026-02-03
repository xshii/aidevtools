"""xlsx 配置管理模块

支持双向转换:
- Python trace → Excel xlsx
- Excel xlsx → Python 代码

使用方式:
    # 生成空模板
    from aidevtools.xlsx import create_template
    create_template("config.xlsx")

    # 从 trace 导出 (保留结果列)
    from aidevtools.xlsx import export_xlsx
    export_xlsx("config.xlsx", records)

    # 从 xlsx 生成 Python 代码
    from aidevtools.xlsx import import_xlsx
    code = import_xlsx("config.xlsx")
"""
from aidevtools.xlsx.export import export_xlsx
from aidevtools.xlsx.import_ import import_xlsx, parse_xlsx
from aidevtools.xlsx.op_registry import (
    get_default_ops,
    get_op_info,
    list_ops,
)
from aidevtools.xlsx.run import run_xlsx
from aidevtools.xlsx.template import create_template

__all__ = [
    "get_default_ops",
    "get_op_info",
    "list_ops",
    "create_template",
    "export_xlsx",
    "import_xlsx",
    "parse_xlsx",
    "run_xlsx",
]
