"""xlsx 运行

从 xlsx 配置运行比对流程。
"""
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
    from aidevtools.ops.base import clear, dump, gen_csv, get_records
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
