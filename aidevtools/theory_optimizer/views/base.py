"""
视图基类

设计模式:
- Template Method 模式: 定义渲染流程
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from enum import Enum


class ViewFormat(Enum):
    """输出格式"""
    TEXT = "text"
    JSON = "json"
    HTML = "html"
    SVG = "svg"


@dataclass
class ViewResult:
    """视图渲染结果"""
    view_type: str
    format: ViewFormat
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def save(self, path: str) -> None:
        """保存到文件"""
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.content)

    def __str__(self) -> str:
        return self.content


class View(ABC):
    """
    视图基类 (Template Method 模式)

    定义渲染流程，子类实现具体渲染逻辑
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """视图名称"""
        pass

    @property
    def description(self) -> str:
        """视图描述"""
        return ""

    @abstractmethod
    def render(self, data: Any, format: ViewFormat = ViewFormat.TEXT) -> ViewResult:
        """
        渲染视图

        Args:
            data: 待渲染的数据
            format: 输出格式

        Returns:
            ViewResult: 渲染结果
        """
        pass

    # Template Method: 定义渲染流程
    def generate(self, data: Any,
                 format: ViewFormat = ViewFormat.TEXT,
                 **options) -> ViewResult:
        """
        生成视图 (Template Method)

        子类可以覆盖各个步骤
        """
        # 1. 预处理数据
        processed_data = self._preprocess(data)

        # 2. 验证数据
        self._validate(processed_data)

        # 3. 渲染
        result = self.render(processed_data, format)

        # 4. 后处理
        result = self._postprocess(result, **options)

        return result

    def _preprocess(self, data: Any) -> Any:
        """预处理 hook"""
        return data

    def _validate(self, data: Any) -> None:
        """验证 hook"""
        pass

    def _postprocess(self, result: ViewResult, **options) -> ViewResult:
        """后处理 hook"""
        return result

    # 辅助方法
    def _format_number(self, num: float, precision: int = 2) -> str:
        """格式化数字"""
        if num >= 1e9:
            return f"{num/1e9:.{precision}f}G"
        elif num >= 1e6:
            return f"{num/1e6:.{precision}f}M"
        elif num >= 1e3:
            return f"{num/1e3:.{precision}f}K"
        return f"{num:.{precision}f}"

    def _format_bytes(self, bytes: int) -> str:
        """格式化字节数"""
        if bytes >= 1024**3:
            return f"{bytes/1024**3:.2f} GB"
        elif bytes >= 1024**2:
            return f"{bytes/1024**2:.2f} MB"
        elif bytes >= 1024:
            return f"{bytes/1024:.2f} KB"
        return f"{bytes} B"

    def _format_cycles(self, cycles: int) -> str:
        """格式化周期数"""
        return self._format_number(cycles)

    def _create_bar(self, value: float, max_value: float,
                   width: int = 40, fill: str = "█", empty: str = "░") -> str:
        """创建进度条"""
        if max_value <= 0:
            return empty * width

        ratio = min(1.0, value / max_value)
        filled = int(width * ratio)
        return fill * filled + empty * (width - filled)


class ViewRegistry:
    """视图注册表"""
    _views: Dict[str, type] = {}

    @classmethod
    def register(cls, name: str):
        """装饰器：注册视图"""
        def decorator(view_class: type):
            cls._views[name] = view_class
            return view_class
        return decorator

    @classmethod
    def get(cls, name: str) -> Optional[type]:
        """获取视图类"""
        return cls._views.get(name)

    @classmethod
    def create(cls, name: str, **kwargs) -> Optional[View]:
        """创建视图实例"""
        view_class = cls.get(name)
        if view_class:
            return view_class(**kwargs)
        return None

    @classmethod
    def list_views(cls) -> List[str]:
        """列出所有视图"""
        return list(cls._views.keys())
