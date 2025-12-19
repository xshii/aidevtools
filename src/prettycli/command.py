from abc import ABC, abstractmethod
from typing import Dict, Type, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from prettycli.context import Context


class BaseCommand(ABC):
    """命令基类，所有命令继承此类"""

    name: str = ""
    help: str = ""

    _registry: Dict[str, Type["BaseCommand"]] = {}

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if cls.name:
            BaseCommand._registry[cls.name] = cls

    @abstractmethod
    def run(self, ctx: "Context", **kwargs) -> int:
        """执行命令，返回 exit code"""
        pass

    @classmethod
    def get(cls, name: str) -> Optional[Type["BaseCommand"]]:
        return cls._registry.get(name)

    @classmethod
    def all(cls) -> Dict[str, Type["BaseCommand"]]:
        return cls._registry.copy()

    @classmethod
    def clear(cls):
        """清空注册（测试用）"""
        cls._registry.clear()
