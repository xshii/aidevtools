"""日志模块"""
import sys
from datetime import datetime
from enum import IntEnum

class Level(IntEnum):
    DEBUG = 10
    INFO = 20
    WARN = 30
    ERROR = 40

DEBUG, INFO, WARN, ERROR = Level.DEBUG, Level.INFO, Level.WARN, Level.ERROR

_level = INFO
_module = "aidevtools"

def set_level(level: Level):
    global _level
    _level = level

def set_module(name: str):
    global _module
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
    def __init__(self, module: str = ""):
        self.module = module or _module

    def debug(self, msg: str):
        _log(DEBUG, self.module, msg)

    def info(self, msg: str):
        _log(INFO, self.module, msg)

    def warn(self, msg: str):
        _log(WARN, self.module, msg)

    def error(self, msg: str):
        _log(ERROR, self.module, msg)

logger = Logger()
