"""测试用例生成器

用于 MIXED 模式，自动生成单算子 + 双算子组合 + 完整图测试用例。

对于算子链 op1 → op2 → op3，自动生成：
├── 单算子测试
│   ├── test_op1_standalone
│   ├── test_op2_standalone
│   └── test_op3_standalone
├── 双算子测试
│   ├── test_op1_op2_chain
│   └── test_op2_op3_chain
└── 完整图测试
    └── test_full_graph

用法:
    import aidevtools.ops as ops
    from aidevtools.ops.base import CompareMode

    ops.set_compare_mode(CompareMode.MIXED)
    ops.clear()

    x = ops.traced(input_data, "gfp16")
    y = F.matmul(x, w)
    y = F.gelu(y)

    test_cases = ops.generate_test_cases()
    results = ops.run_test_cases(test_cases)
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from aidevtools.core.log import logger


@dataclass
class TestCase:
    """测试用例

    Attributes:
        name: 测试用例名（如 "matmul_0_standalone"）
        ops: 包含的算子名列表（如 ["matmul_0"]）
        inputs: 输入数据字典
        expected_output: 期望输出（最后一个算子的 reference）
        test_type: 测试类型 ("standalone", "chain", "full_graph")
    """
    name: str
    ops: List[str]
    inputs: Dict[str, np.ndarray] = field(default_factory=dict)
    expected_output: Optional[np.ndarray] = None
    test_type: str = "standalone"


@dataclass
class TestResult:
    """测试结果

    Attributes:
        name: 测试用例名
        passed: 是否通过
        golden: Golden 输出
        reference: Reference 输出
        max_diff: 最大差异
        mean_diff: 平均差异
        error: 错误信息（如果有）
    """
    name: str
    passed: bool
    golden: Optional[np.ndarray] = None
    reference: Optional[np.ndarray] = None
    max_diff: float = 0.0
    mean_diff: float = 0.0
    error: Optional[str] = None


def generate_test_cases(
    include_standalone: bool = True,
    include_chain: bool = True,
    include_full_graph: bool = True,
) -> List[TestCase]:
    """生成测试用例

    根据当前计算图，自动生成单算子、双算子链、完整图测试用例。

    Args:
        include_standalone: 是否包含单算子测试
        include_chain: 是否包含双算子链测试
        include_full_graph: 是否包含完整图测试

    Returns:
        测试用例列表
    """
    from aidevtools.ops.base import get_graph, get_graph_ops

    graph = get_graph()
    ops_list = get_graph_ops()

    if not ops_list:
        logger.warning("计算图为空，无法生成测试用例")
        return []

    test_cases = []

    # 1. 单算子测试
    if include_standalone:
        for op_name in ops_list:
            node = graph[op_name]
            test_case = TestCase(
                name=f"{op_name}_standalone",
                ops=[op_name],
                inputs=node.input_data.copy(),
                expected_output=node.output_data,
                test_type="standalone",
            )
            test_cases.append(test_case)
            logger.debug(f"生成单算子测试: {test_case.name}")

    # 2. 双算子链测试
    if include_chain and len(ops_list) >= 2:
        for i in range(len(ops_list) - 1):
            op1_name = ops_list[i]
            op2_name = ops_list[i + 1]
            node1 = graph[op1_name]
            node2 = graph[op2_name]

            # 检查是否有依赖关系
            if op1_name in node2.inputs:
                test_case = TestCase(
                    name=f"{op1_name}_{op2_name}_chain",
                    ops=[op1_name, op2_name],
                    inputs=node1.input_data.copy(),  # 使用第一个算子的输入
                    expected_output=node2.output_data,
                    test_type="chain",
                )
                test_cases.append(test_case)
                logger.debug(f"生成双算子链测试: {test_case.name}")

    # 3. 完整图测试
    if include_full_graph and len(ops_list) >= 1:
        first_node = graph[ops_list[0]]
        last_node = graph[ops_list[-1]]
        test_case = TestCase(
            name="full_graph",
            ops=ops_list.copy(),
            inputs=first_node.input_data.copy(),
            expected_output=last_node.output_data,
            test_type="full_graph",
        )
        test_cases.append(test_case)
        logger.debug(f"生成完整图测试: {test_case.name}")

    logger.info(f"共生成 {len(test_cases)} 个测试用例")
    return test_cases


def run_test_cases(
    test_cases: List[TestCase],
    rtol: float = 1e-3,
    atol: float = 1e-5,
) -> List[TestResult]:
    """执行测试用例

    Args:
        test_cases: 测试用例列表
        rtol: 相对容差
        atol: 绝对容差

    Returns:
        测试结果列表
    """

    results = []

    for test_case in test_cases:
        logger.info(f"执行测试: {test_case.name}")

        try:
            result = _run_single_test(test_case, rtol, atol)
            results.append(result)

            if result.passed:
                logger.info(f"  ✓ 通过 (max_diff={result.max_diff:.6f})")
            else:
                logger.warning(f"  ✗ 失败 (max_diff={result.max_diff:.6f})")
                if result.error:
                    logger.warning(f"    错误: {result.error}")

        except Exception as e:
            result = TestResult(
                name=test_case.name,
                passed=False,
                error=str(e),
            )
            results.append(result)
            logger.error(f"  ✗ 异常: {e}")

    # 统计
    passed = sum(1 for r in results if r.passed)
    failed = len(results) - passed
    logger.info(f"测试完成: {passed} 通过, {failed} 失败")

    return results


def _run_single_test(
    test_case: TestCase,
    rtol: float,
    atol: float,
) -> TestResult:
    """执行单个测试用例

    Args:
        test_case: 测试用例
        rtol: 相对容差
        atol: 绝对容差

    Returns:
        测试结果
    """
    from aidevtools.ops.base import get_graph

    graph = get_graph()

    if test_case.test_type == "standalone":
        # 单算子测试：直接比对该算子的 golden 和 reference
        return _run_standalone_test(test_case, graph, rtol, atol)
    elif test_case.test_type == "chain":
        # 双算子链测试：重新执行两个算子并比对最终输出
        return _run_chain_test(test_case, graph, rtol, atol)
    else:
        # 完整图测试：比对最终输出
        return _run_full_graph_test(test_case, graph, rtol, atol)


def _run_standalone_test(
    test_case: TestCase,
    graph: Dict[str, Any],
    rtol: float,
    atol: float,
) -> TestResult:
    """执行单算子测试"""
    from aidevtools.ops.base import get_records

    op_name = test_case.ops[0]

    # 从记录中找到该算子的结果
    records = get_records()
    record = None
    for r in records:
        if r["name"] == op_name:
            record = r
            break

    if record is None:
        return TestResult(
            name=test_case.name,
            passed=False,
            error=f"找不到算子记录: {op_name}",
        )

    golden = record.get("golden")
    reference = record.get("reference")

    if golden is None:
        return TestResult(
            name=test_case.name,
            passed=False,
            error="Golden 输出为空",
        )

    if reference is None:
        # 没有 reference，需要重新计算
        return _recompute_and_compare(test_case, graph, rtol, atol)

    # 比对
    return _compare_arrays(test_case.name, golden, reference, rtol, atol)


def _run_chain_test(
    test_case: TestCase,
    graph: Dict[str, Any],
    rtol: float,
    atol: float,
) -> TestResult:
    """执行双算子链测试"""
    return _recompute_and_compare(test_case, graph, rtol, atol)


def _run_full_graph_test(
    test_case: TestCase,
    graph: Dict[str, Any],
    rtol: float,
    atol: float,
) -> TestResult:
    """执行完整图测试"""
    return _recompute_and_compare(test_case, graph, rtol, atol)


def _recompute_and_compare(
    test_case: TestCase,
    graph: Dict[str, Any],
    rtol: float,
    atol: float,
) -> TestResult:
    """重新计算并比对

    通过重新执行算子链，获取 golden 和 reference 输出，然后比对。
    """
    from aidevtools.ops.base import get_records

    # 获取最后一个算子的结果
    last_op = test_case.ops[-1]
    records = get_records()

    golden = None
    reference = None
    for r in records:
        if r["name"] == last_op:
            golden = r.get("golden")
            reference = r.get("reference")
            break

    if golden is None:
        return TestResult(
            name=test_case.name,
            passed=False,
            error=f"找不到算子 {last_op} 的 golden 输出",
        )

    # 如果没有 reference，使用 expected_output（来自计算图）
    if reference is None:
        reference = test_case.expected_output

    if reference is None:
        return TestResult(
            name=test_case.name,
            passed=False,
            error="无法获取 reference 输出进行比对",
        )

    return _compare_arrays(test_case.name, golden, reference, rtol, atol)


def _compare_arrays(
    name: str,
    golden: np.ndarray,
    reference: np.ndarray,
    rtol: float,
    atol: float,
) -> TestResult:
    """比对两个数组

    Args:
        name: 测试名
        golden: Golden 输出
        reference: Reference 输出
        rtol: 相对容差
        atol: 绝对容差

    Returns:
        测试结果
    """
    golden = np.asarray(golden, dtype=np.float32)
    reference = np.asarray(reference, dtype=np.float32)

    if golden.shape != reference.shape:
        return TestResult(
            name=name,
            passed=False,
            golden=golden,
            reference=reference,
            error=f"形状不匹配: golden={golden.shape}, reference={reference.shape}",
        )

    diff = np.abs(golden - reference)
    max_diff = float(np.max(diff))
    mean_diff = float(np.mean(diff))

    # 使用 np.allclose 判断是否通过
    passed = np.allclose(golden, reference, rtol=rtol, atol=atol)

    return TestResult(
        name=name,
        passed=passed,
        golden=golden,
        reference=reference,
        max_diff=max_diff,
        mean_diff=mean_diff,
    )


def print_test_summary(results: List[TestResult]) -> None:
    """打印测试摘要

    Args:
        results: 测试结果列表
    """
    print("\n" + "=" * 60)
    print("测试摘要")
    print("=" * 60)

    passed = [r for r in results if r.passed]
    failed = [r for r in results if not r.passed]

    print(f"\n总计: {len(results)} 个测试")
    print(f"通过: {len(passed)} ({100*len(passed)/len(results):.1f}%)")
    print(f"失败: {len(failed)} ({100*len(failed)/len(results):.1f}%)")

    if failed:
        print("\n失败的测试:")
        for r in failed:
            print(f"  - {r.name}")
            if r.error:
                print(f"    错误: {r.error}")
            else:
                print(f"    max_diff={r.max_diff:.6f}, mean_diff={r.mean_diff:.6f}")

    print("=" * 60 + "\n")
