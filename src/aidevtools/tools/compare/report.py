"""报告生成"""
import json
from pathlib import Path
from typing import Dict, List

from aidevtools.core.log import logger


def gen_report(op_name: str, diff_result, blocks: List[Dict],
               output_dir: str) -> str:
    """生成文本报告"""
    path = Path(output_dir) / op_name
    path.mkdir(parents=True, exist_ok=True)

    report_file = path / "summary.txt"
    lines = [
        f"算子: {op_name}",
        f"状态: {'PASS' if diff_result.passed else 'FAIL'}",
        "",
        "[完整级指标]",
        f"  max_abs:  {diff_result.max_abs:.6e}",
        f"  mean_abs: {diff_result.mean_abs:.6e}",
        f"  max_rel:  {diff_result.max_rel:.6e}",
        f"  qsnr:     {diff_result.qsnr:.2f} dB",
        f"  cosine:   {diff_result.cosine:.6f}",
        f"  总元素:   {diff_result.total_elements}",
        f"  超阈值:   {diff_result.exceed_count}",
        "",
        "[分块级]",
        f"  总块数:   {len(blocks)}",
        f"  失败块:   {sum(1 for b in blocks if not b['passed'])}",
    ]

    report_file.write_text("\n".join(lines))

    # 保存分块详情
    blocks_file = path / "blocks.json"
    blocks_file.write_text(json.dumps(blocks, indent=2))

    logger.info(f"报告生成: {report_file}")
    return str(path)

def gen_heatmap_svg(blocks: List[Dict], output_path: str,
                    cols: int = 64, cell_size: int = 8):
    """生成 SVG 热力图"""
    rows = (len(blocks) + cols - 1) // cols
    width = cols * cell_size
    height = rows * cell_size

    svg_parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        f'<rect width="{width}" height="{height}" fill="#f0f0f0"/>',
    ]

    for i, block in enumerate(blocks):
        x = (i % cols) * cell_size
        y = (i // cols) * cell_size

        # 颜色：绿=通过，红=失败，深浅按 qsnr
        if block["passed"]:
            color = "#4caf50"  # 绿
        else:
            qsnr = block.get("qsnr", 0)
            if qsnr > 30:
                color = "#ffeb3b"  # 黄
            elif qsnr > 20:
                color = "#ff9800"  # 橙
            else:
                color = "#f44336"  # 红

        svg_parts.append(
            f'<rect x="{x}" y="{y}" width="{cell_size-1}" height="{cell_size-1}" '
            f'fill="{color}" title="offset:{block["offset"]} qsnr:{block.get("qsnr", 0):.1f}"/>'
        )

    svg_parts.append('</svg>')

    Path(output_path).write_text("\n".join(svg_parts), encoding="utf-8")
    logger.info(f"热力图生成: {output_path}")
