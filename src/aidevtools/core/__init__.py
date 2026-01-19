"""核心模块"""
from aidevtools.core.log import logger
from aidevtools.core.config import (
    GlobalConfig,
    ExactConfig,
    FuzzyConfig,
    get_config,
    set_config,
    reset_config,
)
from aidevtools.core.tensor import (
    Tensor,
    generate_random,
    generate_weight,
)
from aidevtools.core.op import (
    OpSpec,
    register_op,
    get_op,
    list_ops,
    run_golden,
)
from aidevtools.core.engine import (
    OpRecord,
    ExecutionEngine,
    get_engine,
    clear,
    run_op,
    get_records,
    dump,
)

__all__ = [
    # config
    "GlobalConfig",
    "ExactConfig",
    "FuzzyConfig",
    "get_config",
    "set_config",
    "reset_config",
    # tensor
    "Tensor",
    "generate_random",
    "generate_weight",
    # op
    "OpSpec",
    "register_op",
    "get_op",
    "list_ops",
    "run_golden",
    # engine
    "OpRecord",
    "ExecutionEngine",
    "get_engine",
    "clear",
    "run_op",
    "get_records",
    "dump",
    # log
    "logger",
]
