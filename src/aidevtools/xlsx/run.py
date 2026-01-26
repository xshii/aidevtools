"""xlsx 运行

从 xlsx 配置运行比对流程。
"""

import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

try:
    import openpyxl  # noqa: F401  # pylint: disable=unused-import

    HAS_OPENPYXL = True
except ImportError:
    HAS_OPENPYXL = False

from aidevtools.core.log import logger
from aidevtools.xlsx.export import update_compare_results
from aidevtools.xlsx.import_ import OpConfig, parse_xlsx


def _run_sim_cmd(
    sim_cmd: str,
    config: "OpConfig",
    output_dir: Path,
    name: str,
    has_input: bool = True,
    has_weight: bool = False,
) -> Optional[np.ndarray]:
    """
    执行仿真命令

    Args:
        sim_cmd: 仿真命令模板，支持占位符
        config: 算子配置（包含自定义路径）
        output_dir: 输出目录
        name: 记录名称 (如 "linear_0")
        has_input: 是否有输入文件
        has_weight: 是否有权重文件

    Returns:
        仿真结果数组，失败返回 None

    占位符:
        {golden_bin} - golden 文件路径
        {result_bin} - result 文件路径
        {input_bin} - 输入文件路径
        {weight_bin} - 权重文件路径
        {id} - 算子 ID
        {op_name} - 算子名称
    """
    if not sim_cmd or not sim_cmd.strip():
        return None

    # 构建路径（优先使用配置中的自定义路径）
    golden_bin = config.paths.golden or str(output_dir / f"{name}_golden.bin")
    result_bin = config.paths.result or str(output_dir / f"{name}_result.bin")
    input_bin = config.paths.input or (str(output_dir / f"{name}_input.bin") if has_input else "")
    weight_bin = config.paths.weight or (
        str(output_dir / f"{name}_weight.bin") if has_weight else ""
    )

    # 替换占位符
    cmd = sim_cmd.format(
        golden_bin=golden_bin,
        result_bin=result_bin,
        input_bin=input_bin,
        weight_bin=weight_bin,
        id=config.id,
        op_name=config.op_name,
    )

    logger.info(f"执行仿真命令: {cmd}")

    try:
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=300,  # 5 分钟超时
            check=False,  # 手动检查 returncode
        )

        if result.returncode != 0:
            logger.error(f"仿真命令失败 (code={result.returncode}): {result.stderr}")
            return None

        if result.stdout:
            logger.debug(f"仿真输出: {result.stdout[:200]}")

        # 检查 result_bin 是否存在
        result_path = Path(result_bin)
        if result_path.exists():
            from aidevtools.formats.base import load as load_data

            data = load_data(result_bin)
            logger.info(f"仿真结果: {result_bin}, shape={data.shape}")
            return data
        logger.warning(f"仿真命令执行完成，但未生成 result 文件: {result_bin}")
        return None

    except subprocess.TimeoutExpired:
        logger.error(f"仿真命令超时 (>300s): {cmd}")
        return None
    except (OSError, subprocess.SubprocessError) as e:
        # OSError: 命令不存在或权限问题
        # SubprocessError: 子进程执行问题
        logger.error(f"仿真命令异常: {e}")
        return None


def _get_binary_paths(
    config: "OpConfig", output_dir: Path, name: str, has_input: bool, has_weight: bool
) -> dict:
    """获取 binary 路径（优先使用配置中的自定义路径）"""
    return {
        "golden_bin": config.paths.golden or str(output_dir / f"{name}_golden.bin"),
        "result_bin": config.paths.result or str(output_dir / f"{name}_result.bin"),
        "input_bin": config.paths.input
        or (str(output_dir / f"{name}_input.bin") if has_input else ""),
        "weight_bin": config.paths.weight
        or (str(output_dir / f"{name}_weight.bin") if has_weight else ""),
    }


def _check_openpyxl():
    if not HAS_OPENPYXL:
        raise ImportError("xlsx 功能需要 openpyxl，请安装: pip install openpyxl")


def _process_single_config(config: OpConfig, enabled_ops: List[str], outputs: Dict, results: List):
    """处理单个算子配置，返回是否成功执行"""
    if config.skip:
        logger.info(f"[SKIP] {config.op_name} (id={config.id})")
        results.append({"id": config.id, "status": "SKIP", "note": "用户跳过"})
        return False

    if enabled_ops and config.op_name not in enabled_ops:
        logger.warning(f"算子 {config.op_name} 未在 op_registry 中启用，跳过")
        results.append({"id": config.id, "status": "SKIP", "note": f"算子 {config.op_name} 未启用"})
        return False

    try:
        inputs = _generate_inputs(config, outputs)
        output = _execute_op(config, inputs)
        outputs[config.id] = output
        logger.debug(f"执行 {config.op_name}_{config.id}: shape={output.shape}")
        return True
    except (ValueError, TypeError, RuntimeError, KeyError, AttributeError) as e:
        logger.error(f"执行 {config.op_name}_{config.id} 失败: {e}")
        results.append({"id": config.id, "status": "ERROR", "note": str(e)})
        return False


def _run_simulations(records: List, op_configs: List[OpConfig], out_path: Path):
    """执行仿真命令"""
    sim_cmd_map = {cfg.op_name: cfg for cfg in op_configs if cfg.paths.sim_cmd}

    for record in records:
        op_type = record.get("op")
        if op_type in sim_cmd_map and record.get("result") is None:
            config = sim_cmd_map[op_type]
            sim_result = _run_sim_cmd(
                sim_cmd=config.paths.sim_cmd,
                config=config,
                output_dir=out_path,
                name=record.get("name", ""),
                has_input=record.get("input") is not None,
                has_weight=record.get("weight") is not None,
            )
            if sim_result is not None:
                record["result"] = sim_result
                logger.info(f"仿真完成: {record.get('name')}, shape={sim_result.shape}")


def _compare_record(record: Dict, idx: int, out_path: Path) -> Dict[str, Any]:
    """比对单条记录"""
    from aidevtools.tools.compare.diff import compare_full, compare_isclose

    golden = record.get("golden")
    result = record.get("result")
    name = record.get("name", f"op_{idx}")

    res = {
        "id": idx,
        "golden_bin": str(out_path / f"{name}_golden.bin"),
        "result_bin": str(out_path / f"{name}_result.bin") if result is not None else "",
    }

    if result is None:
        res["status"] = "PENDING"
        res["note"] = "result 待填充"
        return res

    # compare_full
    diff = compare_full(np.asarray(golden), np.asarray(result))
    res["status"] = "PASS" if diff.passed else "FAIL"
    res["max_abs"] = f"{diff.max_abs:.6e}"
    res["qsnr"] = f"{diff.qsnr:.2f}"
    res["cosine"] = f"{diff.cosine:.6f}"

    # compare_isclose
    isclose = compare_isclose(
        np.asarray(golden), np.asarray(result), atol=1e-4, rtol=1e-2, max_exceed_ratio=0.01
    )
    res["isclose_pass"] = "PASS" if isclose.passed else "FAIL"
    res["exceed_count"] = str(isclose.exceed_count)
    res["exceed_ratio"] = f"{isclose.exceed_ratio:.4%}"

    return res


def run_xlsx(
    xlsx_path: str,
    output_dir: str = "./workspace",
    fmt: str = "raw",
) -> List[Dict[str, Any]]:
    """从 xlsx 配置运行算子并比对"""
    _check_openpyxl()
    from aidevtools.ops.base import clear, dump, get_records
    from aidevtools.xlsx.export import export_xlsx

    enabled_ops, op_configs = parse_xlsx(xlsx_path)
    logger.info(f"加载配置: {len(op_configs)} 个算子")

    clear()
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # 执行算子
    outputs, results = {}, []
    for config in op_configs:
        _process_single_config(config, enabled_ops, outputs, results)

    # 获取记录并导出
    records = get_records()
    dump(output_dir, fmt=fmt)

    # 执行仿真
    _run_simulations(records, op_configs, out_path)

    # 导出到 xlsx
    export_xlsx(xlsx_path, records, preserve_results=True)

    # 比对
    for idx, record in enumerate(records):
        if record.get("golden") is not None:
            results.append(_compare_record(record, idx, out_path))

    # 更新结果
    update_compare_results(xlsx_path, results)

    # 统计
    pass_count = sum(1 for r in results if r.get("status") == "PASS")
    fail_count = sum(1 for r in results if r.get("status") == "FAIL")
    skip_count = sum(1 for r in results if r.get("status") in ("SKIP", "PENDING"))
    logger.info(f"比对完成: PASS={pass_count}, FAIL={fail_count}, SKIP={skip_count}")

    return results


def _generate_inputs(config: OpConfig, outputs: Dict[int, np.ndarray]) -> Dict[str, np.ndarray]:
    """生成算子输入"""
    depends = config.parse_depends()
    dtype = getattr(np, config.dtype, np.float32)

    if not depends:
        # 无依赖，随机输入
        shape = config.shape if config.shape else (1, 64)
        return {"x": np.random.randn(*shape).astype(dtype)}

    # 有依赖
    inputs = {}
    for name, deps in depends.items():
        for dep_id in deps:
            if dep_id in outputs:
                inputs[name] = outputs[dep_id]
            else:
                logger.warning(f"依赖 {dep_id} 不存在，使用随机数据")
                shape = config.shape if config.shape else (1, 64)
                inputs[name] = np.random.randn(*shape).astype(dtype)

    return inputs


def _get_input(inputs: Dict[str, np.ndarray], key: str = "x") -> np.ndarray:
    """获取输入，优先使用指定 key，否则取第一个"""
    return inputs.get(key, list(inputs.values())[0])


def _exec_linear(inputs, config, dtype, F_module):
    x = _get_input(inputs)
    out_features = config.shape[-1] if config.shape else 256
    # PyTorch 格式: weight [out_features, in_features]
    weight = np.random.randn(out_features, x.shape[-1]).astype(dtype)
    return F_module.linear(x, weight)


def _exec_matmul(inputs, _config, dtype, F_module):
    if len(inputs) >= 2:
        keys = list(inputs.keys())
        return F_module.matmul(inputs[keys[0]], inputs[keys[1]])
    x = _get_input(inputs)
    b = np.random.randn(x.shape[-1], x.shape[-1]).astype(dtype)
    return F_module.matmul(x, b)


def _exec_attention(inputs, _config, _dtype, F_module):
    q, k, v = inputs.get("q"), inputs.get("k"), inputs.get("v")
    if q is not None and k is not None and v is not None:
        return F_module.attention(q, k, v)
    x = _get_input(inputs)
    return F_module.attention(x, x, x)


def _exec_binary_op(op_func):
    """创建二元算子执行器"""

    def executor(inputs, _config, _dtype, _F):
        if len(inputs) >= 2:
            keys = list(inputs.keys())
            return op_func(inputs[keys[0]], inputs[keys[1]])
        x = _get_input(inputs)
        return op_func(x, x)

    return executor


def _exec_layernorm(inputs, _config, dtype, F_module):
    x = _get_input(inputs)
    gamma = np.ones(x.shape[-1], dtype=dtype)
    beta = np.zeros(x.shape[-1], dtype=dtype)
    # PyTorch 风格签名: layernorm(input, normalized_shape, weight, bias, eps)
    return F_module.layernorm(x, normalized_shape=(x.shape[-1],), weight=gamma, bias=beta)


def _exec_unary(op_name):
    """创建一元算子执行器"""

    def executor(inputs, _config, _dtype, F_module):
        return getattr(F_module, op_name)(_get_input(inputs))

    return executor


def _execute_op(config: OpConfig, inputs: Dict[str, np.ndarray]) -> np.ndarray:
    """执行单个算子 - 使用分发表"""
    from aidevtools.ops import _functional as F_module

    op_name = config.op_name
    dtype = getattr(np, config.dtype, np.float32)

    # 算子分发表
    executors = {
        "linear": _exec_linear,
        "matmul": _exec_matmul,
        "attention": _exec_attention,
        "add": _exec_binary_op(F_module.add),
        "mul": _exec_binary_op(F_module.mul),
        "layernorm": _exec_layernorm,
        "relu": _exec_unary("relu"),
        "softmax": _exec_unary("softmax"),
        "gelu": _exec_unary("gelu"),
        "sigmoid": _exec_unary("sigmoid"),
        "tanh": _exec_unary("tanh"),
        "silu": _exec_unary("silu"),
    }

    if op_name in executors:
        return executors[op_name](inputs, config, dtype, F_module)

    # 尝试动态调用
    if hasattr(F_module, op_name):
        return getattr(F_module, op_name)(_get_input(inputs))

    raise ValueError(f"未知算子: {op_name}")
