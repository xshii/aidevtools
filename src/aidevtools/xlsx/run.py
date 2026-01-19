"""xlsx 运行

从 xlsx 配置运行比对流程。
"""
import subprocess
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np

try:
    from openpyxl import load_workbook
    HAS_OPENPYXL = True
except ImportError:
    HAS_OPENPYXL = False

from aidevtools.core.log import logger
from aidevtools.xlsx.import_ import parse_xlsx, OpConfig
from aidevtools.xlsx.export import update_compare_results


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
    golden_bin = config.golden_bin or str(output_dir / f"{name}_golden.bin")
    result_bin = config.result_bin or str(output_dir / f"{name}_result.bin")
    input_bin = config.input_bin or (str(output_dir / f"{name}_input.bin") if has_input else "")
    weight_bin = config.weight_bin or (str(output_dir / f"{name}_weight.bin") if has_weight else "")

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
        )

        if result.returncode != 0:
            logger.error(f"仿真命令失败 (code={result.returncode}): {result.stderr}")
            return None

        if result.stdout:
            logger.debug(f"仿真输出: {result.stdout[:200]}")

        # 检查 result_bin 是否存在
        result_path = Path(result_bin)
        if result_path.exists():
            # 读取结果
            from aidevtools.formats.base import load as load_data
            data = load_data(result_bin)
            logger.info(f"仿真结果: {result_bin}, shape={data.shape}")
            return data
        else:
            logger.warn(f"仿真命令执行完成，但未生成 result 文件: {result_bin}")
            return None

    except subprocess.TimeoutExpired:
        logger.error(f"仿真命令超时 (>300s): {cmd}")
        return None
    except Exception as e:
        logger.error(f"仿真命令异常: {e}")
        return None


def _get_binary_paths(config: "OpConfig", output_dir: Path, name: str, has_input: bool, has_weight: bool) -> dict:
    """获取 binary 路径（优先使用配置中的自定义路径）"""
    return {
        "golden_bin": config.golden_bin or str(output_dir / f"{name}_golden.bin"),
        "result_bin": config.result_bin or str(output_dir / f"{name}_result.bin"),
        "input_bin": config.input_bin or (str(output_dir / f"{name}_input.bin") if has_input else ""),
        "weight_bin": config.weight_bin or (str(output_dir / f"{name}_weight.bin") if has_weight else ""),
    }


def _check_openpyxl():
    if not HAS_OPENPYXL:
        raise ImportError("xlsx 功能需要 openpyxl，请安装: pip install openpyxl")


def run_xlsx(
    xlsx_path: str,
    output_dir: str = "./workspace",
    format: str = "raw",
) -> List[Dict[str, Any]]:
    """
    从 xlsx 配置运行算子并比对

    Args:
        xlsx_path: xlsx 文件路径
        output_dir: 输出目录
        format: 数据格式

    Returns:
        比对结果列表
    """
    _check_openpyxl()
    from aidevtools.ops import nn
    from aidevtools.ops.base import clear, dump, get_records
    from aidevtools.tools.compare.diff import compare_full
    from aidevtools.formats.base import save as save_data

    enabled_ops, op_configs = parse_xlsx(xlsx_path)
    logger.info(f"加载配置: {len(op_configs)} 个算子")

    # 清空之前的记录
    clear()

    # 输出目录
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # 执行算子
    outputs = {}
    results = []

    for config in op_configs:
        if config.skip:
            logger.info(f"[SKIP] {config.op_name} (id={config.id})")
            results.append({
                "id": config.id,
                "status": "SKIP",
                "note": "用户跳过",
            })
            continue

        # 检查算子是否启用
        if enabled_ops and config.op_name not in enabled_ops:
            logger.warn(f"算子 {config.op_name} 未在 op_registry 中启用，跳过")
            results.append({
                "id": config.id,
                "status": "SKIP",
                "note": f"算子 {config.op_name} 未启用",
            })
            continue

        try:
            # 生成输入
            inputs = _generate_inputs(config, outputs)

            # 执行算子
            output = _execute_op(config, inputs)
            outputs[config.id] = output

            logger.debug(f"执行 {config.op_name}_{config.id}: shape={output.shape}")

        except Exception as e:
            logger.error(f"执行 {config.op_name}_{config.id} 失败: {e}")
            results.append({
                "id": config.id,
                "status": "ERROR",
                "note": str(e),
            })

    # 获取记录
    records = get_records()

    # 导出数据
    dump(output_dir, format=format)

    # 构建 sim_cmd 映射表 (通过 op_name 索引)
    sim_cmd_map = {}
    for config in op_configs:
        if config.sim_cmd:
            sim_cmd_map[config.op_name] = config

    # 执行仿真命令（如果配置了 sim_cmd）
    for record in records:
        op_type = record.get("op")
        name = record.get("name", "")

        if op_type in sim_cmd_map and record.get("result") is None:
            config = sim_cmd_map[op_type]
            has_input = record.get("input") is not None
            has_weight = record.get("weight") is not None

            sim_result = _run_sim_cmd(
                sim_cmd=config.sim_cmd,
                config=config,
                output_dir=out_path,
                name=name,
                has_input=has_input,
                has_weight=has_weight,
            )

            if sim_result is not None:
                record["result"] = sim_result
                logger.info(f"仿真完成: {name}, shape={sim_result.shape}")

    # 先导出记录到 xlsx（创建 compare 行）
    from aidevtools.xlsx.export import export_xlsx
    export_xlsx(xlsx_path, records, preserve_results=True)

    # 比对并收集结果
    for idx, record in enumerate(records):
        golden = record.get("golden")
        result = record.get("result")

        if golden is None:
            continue

        # 构建文件路径
        name = record.get("name", f"op_{idx}")
        golden_bin = str(out_path / f"{name}_golden.bin")
        result_bin = str(out_path / f"{name}_result.bin") if result is not None else ""

        res = {
            "id": idx,
            "golden_bin": golden_bin,
            "result_bin": result_bin,
        }

        if result is not None:
            # 执行比对
            diff = compare_full(np.asarray(golden), np.asarray(result))
            res["status"] = "PASS" if diff.passed else "FAIL"
            res["max_abs"] = f"{diff.max_abs:.6e}"
            res["qsnr"] = f"{diff.qsnr:.2f}"
            res["cosine"] = f"{diff.cosine:.6f}"
        else:
            res["status"] = "PENDING"
            res["note"] = "result 待填充"

        results.append(res)

    # 更新 xlsx 中的比对结果
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
                logger.warn(f"依赖 {dep_id} 不存在，使用随机数据")
                shape = config.shape if config.shape else (1, 64)
                inputs[name] = np.random.randn(*shape).astype(dtype)

    return inputs


def _execute_op(config: OpConfig, inputs: Dict[str, np.ndarray]) -> np.ndarray:
    """执行单个算子"""
    from aidevtools.ops import nn

    op_name = config.op_name
    dtype = getattr(np, config.dtype, np.float32)

    if op_name == "linear":
        x = inputs.get("x", list(inputs.values())[0])
        in_features = x.shape[-1]
        out_features = config.shape[-1] if config.shape else 256
        weight = np.random.randn(in_features, out_features).astype(dtype)
        return nn.linear(x, weight)

    elif op_name == "matmul":
        if len(inputs) >= 2:
            keys = list(inputs.keys())
            return nn.matmul(inputs[keys[0]], inputs[keys[1]])
        else:
            x = inputs.get("x", list(inputs.values())[0])
            n = x.shape[-1]
            b = np.random.randn(n, n).astype(dtype)
            return nn.matmul(x, b)

    elif op_name == "relu":
        x = inputs.get("x", list(inputs.values())[0])
        return nn.relu(x)

    elif op_name == "softmax":
        x = inputs.get("x", list(inputs.values())[0])
        return nn.softmax(x)

    elif op_name == "attention":
        q = inputs.get("q")
        k = inputs.get("k")
        v = inputs.get("v")
        if q is not None and k is not None and v is not None:
            return nn.attention(q, k, v)
        else:
            # 回退到单输入
            x = list(inputs.values())[0]
            return nn.attention(x, x, x)

    elif op_name == "add":
        if len(inputs) >= 2:
            keys = list(inputs.keys())
            return nn.add(inputs[keys[0]], inputs[keys[1]])
        else:
            x = list(inputs.values())[0]
            return nn.add(x, x)

    elif op_name == "mul":
        if len(inputs) >= 2:
            keys = list(inputs.keys())
            return nn.mul(inputs[keys[0]], inputs[keys[1]])
        else:
            x = list(inputs.values())[0]
            return nn.mul(x, x)

    elif op_name == "layernorm":
        x = inputs.get("x", list(inputs.values())[0])
        gamma = np.ones(x.shape[-1], dtype=dtype)
        beta = np.zeros(x.shape[-1], dtype=dtype)
        return nn.layernorm(x, gamma, beta)

    else:
        # 尝试动态调用
        if hasattr(nn, op_name):
            op_func = getattr(nn, op_name)
            x = inputs.get("x", list(inputs.values())[0])
            return op_func(x)
        else:
            raise ValueError(f"未知算子: {op_name}")
