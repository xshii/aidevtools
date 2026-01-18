"""xlsx 导出

从 trace 记录导出到 xlsx，保留已有的结果列。
"""
from pathlib import Path
from typing import List, Dict, Any, Optional

try:
    from openpyxl import Workbook, load_workbook
    from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
    from openpyxl.chart import PieChart, Reference
    from openpyxl.chart.label import DataLabelList
    HAS_OPENPYXL = True
except ImportError:
    HAS_OPENPYXL = False

from aidevtools.core.log import logger


def _check_openpyxl():
    if not HAS_OPENPYXL:
        raise ImportError("xlsx 功能需要 openpyxl，请安装: pip install openpyxl")


def export_xlsx(
    output_path: str,
    records: List[Dict[str, Any]],
    preserve_results: bool = True,
) -> str:
    """
    导出 trace 记录到 xlsx

    Args:
        output_path: 输出文件路径
        records: trace 记录列表
        preserve_results: 是否保留已有的结果列 (compare sheet)

    Returns:
        输出文件路径

    重要:
        - 如果文件已存在且 preserve_results=True，会保留 compare sheet 中的结果数据
        - ops sheet 会被更新为最新的记录
    """
    _check_openpyxl()

    output_path = Path(output_path)
    existing_compare_data = {}

    # 读取已有的 compare 结果
    if preserve_results and output_path.exists():
        try:
            existing_wb = load_workbook(output_path)
            if "compare" in existing_wb.sheetnames:
                ws = existing_wb["compare"]
                headers = [cell.value for cell in ws[1]]
                for row in ws.iter_rows(min_row=2, values_only=True):
                    row_dict = dict(zip(headers, row))
                    if row_dict.get("id") is not None:
                        existing_compare_data[row_dict["id"]] = row_dict
                logger.debug(f"保留 {len(existing_compare_data)} 条已有比对结果")
            existing_wb.close()
        except Exception as e:
            logger.warn(f"读取已有结果失败: {e}")

    # 创建或加载工作簿
    if output_path.exists():
        wb = load_workbook(output_path)
    else:
        from aidevtools.xlsx.template import create_template
        create_template(str(output_path), include_examples=False)
        wb = load_workbook(output_path)

    # 样式
    header_font = Font(bold=True, color="FFFFFF")
    header_fill = PatternFill(start_color="4472C4", end_color="4472C4", fill_type="solid")
    thin_border = Border(
        left=Side(style='thin'),
        right=Side(style='thin'),
        top=Side(style='thin'),
        bottom=Side(style='thin')
    )

    # ==================== 更新 ops sheet ====================
    if "ops" in wb.sheetnames:
        ws_ops = wb["ops"]
        # 清空数据行 (保留表头)
        ws_ops.delete_rows(2, ws_ops.max_row)
    else:
        ws_ops = wb.create_sheet("ops")
        ops_headers = ["id", "op_name", "shape", "dtype", "depends", "qtype", "skip", "note"]
        for col, header in enumerate(ops_headers, 1):
            cell = ws_ops.cell(row=1, column=col, value=header)
            cell.font = header_font
            cell.fill = header_fill
            cell.border = thin_border

    # 写入记录
    import numpy as np
    for idx, record in enumerate(records):
        row = idx + 2
        name = record.get("name", f"op_{idx}")
        op = record.get("op", name.rsplit("_", 1)[0])

        # 获取 shape 和 dtype
        output = record.get("golden")
        if output is None:
            output = record.get("output")
        if output is not None and hasattr(output, 'shape'):
            arr = np.asarray(output)
            shape_str = ",".join(str(d) for d in arr.shape)
            dtype_str = str(arr.dtype)
        else:
            shape_str = ""
            dtype_str = ""

        ws_ops.cell(row=row, column=1, value=idx).border = thin_border
        ws_ops.cell(row=row, column=2, value=op).border = thin_border
        ws_ops.cell(row=row, column=3, value=shape_str).border = thin_border
        ws_ops.cell(row=row, column=4, value=dtype_str).border = thin_border
        ws_ops.cell(row=row, column=5, value="").border = thin_border  # depends 由用户填写
        ws_ops.cell(row=row, column=6, value="").border = thin_border  # qtype
        ws_ops.cell(row=row, column=7, value="FALSE").border = thin_border  # skip
        ws_ops.cell(row=row, column=8, value=name).border = thin_border  # note 中放 full name

    # ==================== 更新 compare sheet (保留结果) ====================
    if "compare" in wb.sheetnames:
        ws_compare = wb["compare"]
        ws_compare.delete_rows(2, ws_compare.max_row)
    else:
        ws_compare = wb.create_sheet("compare")
        compare_headers = ["id", "op_name", "status", "max_abs", "qsnr", "cosine", "golden_bin", "result_bin", "note"]
        for col, header in enumerate(compare_headers, 1):
            cell = ws_compare.cell(row=1, column=col, value=header)
            cell.font = header_font
            cell.fill = header_fill
            cell.border = thin_border

    # 写入 compare 数据 (合并已有结果)
    for idx, record in enumerate(records):
        row = idx + 2
        name = record.get("name", f"op_{idx}")

        # 检查是否有已保存的结果
        existing = existing_compare_data.get(idx, {})

        ws_compare.cell(row=row, column=1, value=idx).border = thin_border
        ws_compare.cell(row=row, column=2, value=name).border = thin_border

        # 保留已有结果，否则留空
        ws_compare.cell(row=row, column=3, value=existing.get("status", "")).border = thin_border
        ws_compare.cell(row=row, column=4, value=existing.get("max_abs", "")).border = thin_border
        ws_compare.cell(row=row, column=5, value=existing.get("qsnr", "")).border = thin_border
        ws_compare.cell(row=row, column=6, value=existing.get("cosine", "")).border = thin_border
        ws_compare.cell(row=row, column=7, value=existing.get("golden_bin", "")).border = thin_border
        ws_compare.cell(row=row, column=8, value=existing.get("result_bin", "")).border = thin_border
        ws_compare.cell(row=row, column=9, value=existing.get("note", "")).border = thin_border

    # 保存
    wb.save(output_path)
    logger.info(f"导出 xlsx: {output_path} ({len(records)} 条记录)")
    return str(output_path)


def update_compare_results(
    xlsx_path: str,
    results: List[Dict[str, Any]],
) -> str:
    """
    更新 xlsx 中的比对结果

    Args:
        xlsx_path: xlsx 文件路径
        results: 比对结果列表，每项包含 id, status, max_abs, qsnr, cosine 等

    Returns:
        xlsx 文件路径
    """
    _check_openpyxl()

    wb = load_workbook(xlsx_path)
    if "compare" not in wb.sheetnames:
        raise ValueError(f"xlsx 文件缺少 compare sheet: {xlsx_path}")

    ws = wb["compare"]

    # 读取表头
    headers = [cell.value for cell in ws[1]]
    col_map = {h: i + 1 for i, h in enumerate(headers)}

    # 构建 id -> row 映射
    id_to_row = {}
    for row in range(2, ws.max_row + 1):
        cell_id = ws.cell(row=row, column=col_map.get("id", 1)).value
        if cell_id is not None:
            id_to_row[cell_id] = row

    # 结果样式
    pass_fill = PatternFill(start_color="C6EFCE", end_color="C6EFCE", fill_type="solid")
    fail_fill = PatternFill(start_color="FFC7CE", end_color="FFC7CE", fill_type="solid")

    # 更新结果
    for res in results:
        res_id = res.get("id")
        if res_id not in id_to_row:
            continue

        row = id_to_row[res_id]
        status = res.get("status", "")

        if "status" in col_map:
            cell = ws.cell(row=row, column=col_map["status"], value=status)
            if status == "PASS":
                cell.fill = pass_fill
            elif status == "FAIL":
                cell.fill = fail_fill

        if "max_abs" in col_map and "max_abs" in res:
            ws.cell(row=row, column=col_map["max_abs"], value=res["max_abs"])
        if "qsnr" in col_map and "qsnr" in res:
            ws.cell(row=row, column=col_map["qsnr"], value=res["qsnr"])
        if "cosine" in col_map and "cosine" in res:
            ws.cell(row=row, column=col_map["cosine"], value=res["cosine"])
        if "golden_bin" in col_map and "golden_bin" in res:
            ws.cell(row=row, column=col_map["golden_bin"], value=res["golden_bin"])
        if "result_bin" in col_map and "result_bin" in res:
            ws.cell(row=row, column=col_map["result_bin"], value=res["result_bin"])
        if "note" in col_map and "note" in res:
            ws.cell(row=row, column=col_map["note"], value=res["note"])

    # 添加汇总统计
    _add_summary_sheet(wb, results)

    wb.save(xlsx_path)
    logger.info(f"更新比对结果: {xlsx_path}")
    return xlsx_path


def _add_summary_sheet(wb: "Workbook", results: List[Dict[str, Any]]):
    """
    添加汇总统计 sheet

    包含：
    - PASS/FAIL/SKIP/PENDING/ERROR 计数
    - 通过率
    - 饼图
    """
    # 统计
    stats = {"PASS": 0, "FAIL": 0, "SKIP": 0, "PENDING": 0, "ERROR": 0}
    for res in results:
        status = res.get("status", "PENDING")
        if status in stats:
            stats[status] += 1
        else:
            stats["ERROR"] += 1

    total = sum(stats.values())
    pass_rate = (stats["PASS"] / total * 100) if total > 0 else 0

    # 创建或更新 summary sheet
    if "summary" in wb.sheetnames:
        ws = wb["summary"]
        # 清空内容
        for row in ws.iter_rows():
            for cell in row:
                cell.value = None
    else:
        ws = wb.create_sheet("summary", 0)  # 放在最前面

    # 样式
    header_font = Font(bold=True, color="FFFFFF")
    header_fill = PatternFill(start_color="4472C4", end_color="4472C4", fill_type="solid")
    pass_fill = PatternFill(start_color="C6EFCE", end_color="C6EFCE", fill_type="solid")
    fail_fill = PatternFill(start_color="FFC7CE", end_color="FFC7CE", fill_type="solid")
    skip_fill = PatternFill(start_color="FFEB9C", end_color="FFEB9C", fill_type="solid")
    title_font = Font(bold=True, size=14)
    thin_border = Border(
        left=Side(style='thin'),
        right=Side(style='thin'),
        top=Side(style='thin'),
        bottom=Side(style='thin')
    )

    # 标题
    ws.cell(row=1, column=1, value="比对结果汇总").font = title_font
    ws.merge_cells("A1:C1")

    # 统计表格
    ws.cell(row=3, column=1, value="状态").font = header_font
    ws.cell(row=3, column=1).fill = header_fill
    ws.cell(row=3, column=1).border = thin_border
    ws.cell(row=3, column=2, value="数量").font = header_font
    ws.cell(row=3, column=2).fill = header_fill
    ws.cell(row=3, column=2).border = thin_border
    ws.cell(row=3, column=3, value="占比").font = header_font
    ws.cell(row=3, column=3).fill = header_fill
    ws.cell(row=3, column=3).border = thin_border

    row_idx = 4
    status_fills = {
        "PASS": pass_fill,
        "FAIL": fail_fill,
        "SKIP": skip_fill,
        "PENDING": skip_fill,
        "ERROR": fail_fill,
    }

    for status, count in stats.items():
        if count == 0:
            continue
        pct = count / total * 100 if total > 0 else 0

        cell_status = ws.cell(row=row_idx, column=1, value=status)
        cell_status.border = thin_border
        if status in status_fills:
            cell_status.fill = status_fills[status]

        ws.cell(row=row_idx, column=2, value=count).border = thin_border
        ws.cell(row=row_idx, column=3, value=f"{pct:.1f}%").border = thin_border
        row_idx += 1

    # 总计行
    ws.cell(row=row_idx, column=1, value="总计").font = Font(bold=True)
    ws.cell(row=row_idx, column=1).border = thin_border
    ws.cell(row=row_idx, column=2, value=total).font = Font(bold=True)
    ws.cell(row=row_idx, column=2).border = thin_border
    ws.cell(row=row_idx, column=3, value="100%").font = Font(bold=True)
    ws.cell(row=row_idx, column=3).border = thin_border

    # 通过率
    row_idx += 2
    ws.cell(row=row_idx, column=1, value="通过率").font = Font(bold=True, size=12)
    rate_cell = ws.cell(row=row_idx, column=2, value=f"{pass_rate:.1f}%")
    rate_cell.font = Font(bold=True, size=12)
    if pass_rate >= 90:
        rate_cell.fill = pass_fill
    elif pass_rate >= 60:
        rate_cell.fill = skip_fill
    else:
        rate_cell.fill = fail_fill

    # 饼图（只有当有数据时）
    if total > 0:
        # 准备饼图数据（在 E 列）
        chart_data_row = 3
        ws.cell(row=chart_data_row, column=5, value="状态")
        ws.cell(row=chart_data_row, column=6, value="数量")
        chart_row = chart_data_row + 1
        for status, count in stats.items():
            if count > 0:
                ws.cell(row=chart_row, column=5, value=status)
                ws.cell(row=chart_row, column=6, value=count)
                chart_row += 1

        # 创建饼图
        if chart_row > chart_data_row + 1:  # 至少有一行数据
            pie = PieChart()
            pie.title = "比对结果分布"
            labels = Reference(ws, min_col=5, min_row=chart_data_row + 1, max_row=chart_row - 1)
            data = Reference(ws, min_col=6, min_row=chart_data_row, max_row=chart_row - 1)
            pie.add_data(data, titles_from_data=True)
            pie.set_categories(labels)

            # 显示百分比标签
            pie.dataLabels = DataLabelList()
            pie.dataLabels.showPercent = True
            pie.dataLabels.showVal = False
            pie.dataLabels.showCatName = True

            pie.width = 12
            pie.height = 8
            ws.add_chart(pie, "A" + str(row_idx + 3))

    # 列宽
    ws.column_dimensions['A'].width = 12
    ws.column_dimensions['B'].width = 10
    ws.column_dimensions['C'].width = 10
    ws.column_dimensions['E'].width = 10
    ws.column_dimensions['F'].width = 10
