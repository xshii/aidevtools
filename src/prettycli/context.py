from dataclasses import dataclass, field
from typing import Dict, Any
from pathlib import Path


@dataclass
class Context:
    """命令执行上下文"""

    cwd: Path = field(default_factory=Path.cwd)
    verbose: bool = False
    config: Dict[str, Any] = field(default_factory=dict)

    def get_config(self, key: str, default: Any = None) -> Any:
        return self.config.get(key, default)

    def set_config(self, key: str, value: Any):
        self.config[key] = value
