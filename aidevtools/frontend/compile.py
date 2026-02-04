"""
编译封装

封装 py2c 和 c2dut 编译器的调用。

流程:
    Python 路径: model.py -> [py2c] -> model_gen.c + golden/ -> [c2dut] -> model.bin
    C 路径:      model.c -> [c2dut] -> model.bin
"""

import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Union

from aidevtools.core.log import logger
from aidevtools.toolchain import get_compiler


class CompileError(Exception):
    """编译错误"""


class Compiler:
    """
    编译器封装

    使用示例:
        compiler = Compiler()

        # Python 路径
        result = compiler.compile_python(
            source="model.py",
            output="build/model.bin",
            golden_dir="golden/",
        )

        # C 路径
        result = compiler.compile_c(
            source="model.c",
            output="build/model.bin",
        )
    """

    def __init__(
        self,
        py2c_version: Optional[str] = None,
        c2dut_version: Optional[str] = None,
        verbose: bool = False,
    ):
        """
        Args:
            py2c_version: py2c 版本 (可选)
            c2dut_version: c2dut 版本 (可选)
            verbose: 是否输出详细日志
        """
        self.py2c_version = py2c_version
        self.c2dut_version = c2dut_version
        self.verbose = verbose
        self._py2c_path: Optional[str] = None
        self._c2dut_path: Optional[str] = None

    @property
    def py2c_path(self) -> str:
        """获取 py2c 路径 (延迟加载)"""
        if self._py2c_path is None:
            self._py2c_path = get_compiler("py2c", self.py2c_version)
        return self._py2c_path

    @property
    def c2dut_path(self) -> str:
        """获取 c2dut 路径 (延迟加载)"""
        if self._c2dut_path is None:
            self._c2dut_path = get_compiler("c2dut", self.c2dut_version)
        return self._c2dut_path

    def compile_python(
        self,
        source: Union[str, Path],
        output: Union[str, Path],
        golden_dir: Optional[Union[str, Path]] = None,
        options: Optional[Dict] = None,
    ) -> Dict:
        """
        编译 Python 模型

        流程: model.py -> [py2c] -> model_gen.c + golden/ -> [c2dut] -> model.bin

        Args:
            source: Python 源文件
            output: 输出二进制路径
            golden_dir: Golden 数据目录
            options: 额外选项

        Returns:
            编译结果字典
        """
        source = Path(source)
        output = Path(output)

        if golden_dir is None:
            golden_dir = output.parent / "golden"
        golden_dir = Path(golden_dir)

        if not source.exists():
            raise CompileError(f"Source file not found: {source}")

        # 创建输出目录
        output.parent.mkdir(parents=True, exist_ok=True)
        golden_dir.mkdir(parents=True, exist_ok=True)

        # 中间文件
        gen_c = output.parent / f"{source.stem}_gen.c"

        # Step 1: py2c 转换
        logger.info(f"[py2c] {source} -> {gen_c}")
        py2c_result = self._run_py2c(source, gen_c, golden_dir, options)

        # Step 2: c2dut 编译
        logger.info(f"[c2dut] {gen_c} -> {output}")
        c2dut_result = self._run_c2dut(gen_c, output, options)

        return {
            "success": True,
            "source": str(source),
            "output": str(output),
            "golden_dir": str(golden_dir),
            "gen_c": str(gen_c),
            "py2c": py2c_result,
            "c2dut": c2dut_result,
        }

    def compile_c(
        self,
        source: Union[str, Path],
        output: Union[str, Path],
        options: Optional[Dict] = None,
    ) -> Dict:
        """
        编译 C 模型

        流程: model.c -> [c2dut] -> model.bin

        Args:
            source: C 源文件
            output: 输出二进制路径
            options: 额外选项

        Returns:
            编译结果字典
        """
        source = Path(source)
        output = Path(output)

        if not source.exists():
            raise CompileError(f"Source file not found: {source}")

        # 创建输出目录
        output.parent.mkdir(parents=True, exist_ok=True)

        # c2dut 编译
        logger.info(f"[c2dut] {source} -> {output}")
        c2dut_result = self._run_c2dut(source, output, options)

        return {
            "success": True,
            "source": str(source),
            "output": str(output),
            "c2dut": c2dut_result,
        }

    def _run_py2c(
        self,
        source: Path,
        output: Path,
        golden_dir: Path,
        options: Optional[Dict],
    ) -> Dict:
        """运行 py2c"""
        cmd = [
            self.py2c_path,
            str(source),
            "-o",
            str(output),
            "--golden-dir",
            str(golden_dir),
        ]

        if options:
            for key, value in options.get("py2c", {}).items():
                cmd.extend([f"--{key}", str(value)])

        return self._run_cmd(cmd, "py2c")

    def _run_c2dut(
        self,
        source: Path,
        output: Path,
        options: Optional[Dict],
    ) -> Dict:
        """运行 c2dut"""
        cmd = [
            self.c2dut_path,
            str(source),
            "-o",
            str(output),
        ]

        if options:
            for key, value in options.get("c2dut", {}).items():
                cmd.extend([f"--{key}", str(value)])

        return self._run_cmd(cmd, "c2dut")

    def _run_cmd(self, cmd: List[str], name: str) -> Dict:
        """运行命令"""
        if self.verbose:
            logger.debug(f"[{name}] Running: {' '.join(cmd)}")

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,  # 5 分钟超时
                check=False,  # 手动检查返回码
            )

            if result.returncode != 0:
                raise CompileError(
                    f"{name} failed with code {result.returncode}:\n{result.stderr}"
                )

            return {
                "success": True,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "returncode": result.returncode,
            }

        except subprocess.TimeoutExpired as e:
            raise CompileError(f"{name} timed out") from e
        except FileNotFoundError as e:
            raise CompileError(f"{name} not found: {cmd[0]}") from e


def compile_to_dut(
    source: Union[str, Path],
    output: Union[str, Path],
    golden_dir: Optional[Union[str, Path]] = None,
    options: Optional[Dict] = None,
) -> Dict:
    """
    便捷函数: 编译到 DUT

    自动判断 Python 或 C 路径。

    Args:
        source: 源文件路径
        output: 输出二进制路径
        golden_dir: Golden 数据目录 (仅 Python 路径)
        options: 额外选项

    Returns:
        编译结果字典
    """
    source = Path(source)
    compiler = Compiler()

    if source.suffix == ".py":
        return compiler.compile_python(source, output, golden_dir, options)
    if source.suffix in (".c", ".cc", ".cpp"):
        return compiler.compile_c(source, output, options)
    raise CompileError(f"Unsupported source file type: {source.suffix}")
