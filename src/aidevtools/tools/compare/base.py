"""Step 基类"""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class StepResult:
    """Step 执行结果"""
    success: bool
    output_dir: Optional[str] = None
    message: str = ""
    data: Dict[str, Any] = field(default_factory=dict)


class StepBase:
    """Step 基类，所有 step 继承此类"""

    name: str = ""
    desc: str = ""
    input_dir: Optional[str] = None   # 依赖的上一步输出目录
    output_dir: Optional[str] = None  # 本步骤输出目录

    def __init__(self, workspace: str = "./workspace"):
        self.workspace = Path(workspace)

    def get_input_path(self) -> Optional[Path]:
        """获取输入目录路径"""
        if self.input_dir:
            return self.workspace / self.input_dir
        return None

    def get_output_path(self) -> Path:
        """获取输出目录路径"""
        path = self.workspace / self.output_dir
        path.mkdir(parents=True, exist_ok=True)
        return path

    def check_input(self) -> bool:
        """检查输入是否存在"""
        if self.input_dir is None:
            return True
        input_path = self.get_input_path()
        return input_path.exists()

    def run(self, **kwargs) -> StepResult:
        """执行 step，子类实现"""
        raise NotImplementedError
