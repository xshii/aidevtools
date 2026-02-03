"""日志模块"""
import sys
from datetime import datetime
from enum import IntEnum


class Level(IntEnum):
    """日志级别枚举"""
    DEBUG = 10
    INFO = 20
    WARN = 30
    ERROR = 40

DEBUG, INFO, WARN, ERROR = Level.DEBUG, Level.INFO, Level.WARN, Level.ERROR

_level = INFO
_module = "aidevtools"

def set_level(level: Level):
    """设置日志级别"""
    global _level  # pylint: disable=global-statement
    _level = level

def set_module(name: str):
    """设置默认模块名"""
    global _module  # pylint: disable=global-statement
    _module = name

def _log(level: Level, module: str, msg: str):
    if level < _level:
        return
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    name = level.name
    line = f"[{ts}] [{name}] [{module}] {msg}"
    out = sys.stderr if level >= WARN else sys.stdout
    print(line, file=out)

class Logger:
    """日志记录器"""
    def __init__(self, module: str = ""):
        self.module = module or _module

    def debug(self, msg: str):
        """记录 DEBUG 级别日志"""
        _log(DEBUG, self.module, msg)

    def info(self, msg: str):
        """记录 INFO 级别日志"""
        _log(INFO, self.module, msg)

    def warn(self, msg: str):
        """记录 WARN 级别日志"""
        _log(WARN, self.module, msg)

    warning = warn  # 标准 logging 兼容别名

    def error(self, msg: str):
        """记录 ERROR 级别日志"""
        _log(ERROR, self.module, msg)

logger = Logger()
