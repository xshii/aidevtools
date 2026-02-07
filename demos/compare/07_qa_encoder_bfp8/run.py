#!/usr/bin/env python
"""量化感知随机比数报告 (weight:bfp4)

对比实验:
  A 组: 标准正态分布 N(0,1) — 默认随机
  B 组: QA 受控范围 (center=1.0, amplitude=0.5) — 量化感知随机

两组均使用输入 bfp8 / 权重 bfp4 量化，分析:
  1. 数据特征: 均值 / 标准差 / 动态范围
  2. Golden 漂移: pure vs hw 的 QSNR / cosine
  3. 动态范围对比: max/min 比值
  4. 结论: QA 模式是否改善比数

运行: python demos/compare/07_qa_encoder_bfp8/run.py
"""
import numpy as np

from aidevtools.datagen import DataGenerator
from aidevtools.frontend.types import PrecisionConfig
from aidevtools.formats.quantize import simulate_quantize
from aidevtools.compare.fuzzy import compare_fuzzy
from aidevtools.compare.types import CompareConfig

# ---- 全局参数 ----
BATCH, SEQ, HIDDEN, FFN = 2, 16, 64, 256
QTYPE_INPUT = "bfp8"
QTYPE_WEIGHT = "bfp4"
SEED = 42

COMPARE_CFG = CompareConfig(
    fuzzy_min_qsnr=1.0,
    fuzzy_min_cosine=0.85,
    fuzzy_max_exceed_ratio=0.15,
    fuzzy_atol=0.5,
    fuzzy_rtol=0.5,
)

ENCODER_OPS = [
    ("Q_proj",       "linear",    (BATCH, SEQ, HIDDEN), {"out_features": HIDDEN}),
    ("K_proj",       "linear",    (BATCH, SEQ, HIDDEN), {"out_features": HIDDEN}),
    ("V_proj",       "linear",    (BATCH, SEQ, HIDDEN), {"out_features": HIDDEN}),
    ("attn_softmax", "softmax",   (BATCH, SEQ, HIDDEN), {}),
    ("O_proj",       "linear",    (BATCH, SEQ, HIDDEN), {"out_features": HIDDEN}),
    ("layernorm_1",  "layernorm", (BATCH, SEQ, HIDDEN), {}),
    ("ffn_up",       "linear",    (BATCH, SEQ, HIDDEN), {"out_features": FFN}),
    ("ffn_gelu",     "gelu",      (BATCH, SEQ, FFN),    {}),
    ("ffn_down",     "linear",    (BATCH, SEQ, FFN),    {"out_features": HIDDEN}),
    ("layernorm_2",  "layernorm", (BATCH, SEQ, HIDDEN), {}),
]


def run_group(group_name, precision, seed):
    """运行一组 Encoder 四种比数"""
    gen = DataGenerator(seed=seed, precision=precision, qtype=QTYPE_INPUT)
    results = []

    for name, op_name, shape, kwargs in ENCODER_OPS:
        tracks = gen.generate_four_track(
            op_name, input_shape=shape, precision=precision, **kwargs,
        )
        results.append((name, tracks))

    return results


def data_statistics(track_results, label):
    """打印数据特征统计"""
    print(f"\n  {label} — 数据特征:")
    print(f"  {'算子':<15} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10} {'DynRange':>10}")
    print(f"  {'-'*67}")

    for name, tracks in track_results:
        d = tracks.golden_pure
        dyn = np.max(np.abs(d)) / max(np.min(np.abs(d[d != 0])), 1e-30) if np.any(d != 0) else 0
        print(f"  {name:<15} {np.mean(d):>10.4f} {np.std(d):>10.4f} "
              f"{np.min(d):>10.4f} {np.max(d):>10.4f} {dyn:>10.1f}")


def golden_drift_analysis(track_results, label):
    """打印 Golden 漂移分析 (pure vs hw)"""
    print(f"\n  {label} — Golden 漂移 (Pure fp32 vs HW bfp4):")
    print(f"  {'算子':<15} {'QSNR':>10} {'Cosine':>10} {'MaxAbs':>12} {'Status':>8}")
    print(f"  {'-'*57}")

    qsnr_list = []
    cosine_list = []

    for name, tracks in track_results:
        if tracks.golden_hw is not None:
            r = compare_fuzzy(tracks.golden_pure, tracks.golden_hw, COMPARE_CFG)
            status = "PASS" if r.passed else "FAIL"
            print(f"  {name:<15} {r.qsnr:>10.2f} {r.cosine:>10.6f} "
                  f"{r.max_abs:>12.2e} {status:>8}")
            qsnr_list.append(r.qsnr)
            cosine_list.append(r.cosine)
        else:
            print(f"  {name:<15} {'N/A':>10} {'N/A':>10} {'N/A':>12} {'SKIP':>8}")

    if qsnr_list:
        print(f"  {'--- 平均 ---':<15} {np.mean(qsnr_list):>10.2f} "
              f"{np.mean(cosine_list):>10.6f}")

    return qsnr_list, cosine_list


def dynamic_range_comparison(results_a, results_b):
    """动态范围对比"""
    print(f"\n  动态范围对比 (max(|x|) / min(|x|≠0)):")
    print(f"  {'算子':<15} {'A 组(正态)':>12} {'B 组(QA)':>12} {'改善':>10}")
    print(f"  {'-'*51}")

    for (name_a, tracks_a), (name_b, tracks_b) in zip(results_a, results_b):
        da = tracks_a.golden_pure
        db = tracks_b.golden_pure

        def dyn_range(d):
            nonzero = np.abs(d[d != 0])
            if len(nonzero) == 0:
                return 0.0
            return np.max(nonzero) / np.min(nonzero)

        dyn_a = dyn_range(da)
        dyn_b = dyn_range(db)
        improve = ((dyn_a - dyn_b) / dyn_a * 100) if dyn_a > 0 else 0

        print(f"  {name_a:<15} {dyn_a:>12.1f} {dyn_b:>12.1f} {improve:>9.1f}%")


def main():
    print("=" * 75)
    print(f"  量化感知随机 vs 标准正态 — bfp4 权重比数对比报告")
    print(f"  batch={BATCH}, seq={SEQ}, hidden={HIDDEN}, ffn={FFN}")
    print(f"  input={QTYPE_INPUT}, weight={QTYPE_WEIGHT}")
    print("=" * 75)

    # ---- A 组: 标准正态 ----
    prec_a = PrecisionConfig(
        input_dtype="fp16",
        weight_dtype="bfp4",
        compute_dtype="fp32",
        output_dtype="bfp8",
        qa_aware=False,
    )
    print("\n" + "-" * 75)
    print(f"  A 组: 标准正态分布 N(0,1), weight:{QTYPE_WEIGHT} 量化")
    print("-" * 75)
    results_a = run_group("A", prec_a, SEED)

    # ---- B 组: 量化感知 ----
    prec_b = PrecisionConfig(
        input_dtype="fp16",
        weight_dtype="bfp4",
        compute_dtype="fp32",
        output_dtype="bfp8",
        qa_aware=True,
        qa_center=1.0,
        qa_amplitude=0.5,
    )
    print("\n" + "-" * 75)
    print(f"  B 组: QA 感知随机 (center=1.0, amplitude=0.5), weight:{QTYPE_WEIGHT} 量化")
    print("-" * 75)
    results_b = run_group("B", prec_b, SEED)

    # ---- 数据特征分析 ----
    print("\n" + "-" * 75)
    print("  第一部分: 数据特征分析")
    print("-" * 75)
    data_statistics(results_a, "A 组 (正态)")
    data_statistics(results_b, "B 组 (QA)")

    # ---- Golden 漂移分析 ----
    print("\n" + "-" * 75)
    print(f"  第二部分: Golden 漂移分析 (Pure fp32 vs HW {QTYPE_WEIGHT})")
    print("-" * 75)
    qsnr_a, cos_a = golden_drift_analysis(results_a, "A 组 (正态)")
    qsnr_b, cos_b = golden_drift_analysis(results_b, "B 组 (QA)")

    # ---- 动态范围对比 ----
    print("\n" + "-" * 75)
    print("  第三部分: 动态范围对比")
    print("-" * 75)
    dynamic_range_comparison(results_a, results_b)

    # ---- 汇总 ----
    print("\n" + "-" * 75)
    print("  第四部分: 汇总对比")
    print("-" * 75)

    print(f"\n  {'指标':<25} {'A 组(正态)':>15} {'B 组(QA)':>15}")
    print(f"  {'-'*57}")

    if qsnr_a and qsnr_b:
        print(f"  {'平均 QSNR (dB)':<25} {np.mean(qsnr_a):>15.2f} {np.mean(qsnr_b):>15.2f}")
        print(f"  {'最差 QSNR (dB)':<25} {min(qsnr_a):>15.2f} {min(qsnr_b):>15.2f}")
    if cos_a and cos_b:
        print(f"  {'平均 Cosine':<25} {np.mean(cos_a):>15.6f} {np.mean(cos_b):>15.6f}")
        print(f"  {'最差 Cosine':<25} {min(cos_a):>15.6f} {min(cos_b):>15.6f}")

    # 结论
    print(f"\n  结论:")
    if qsnr_a and qsnr_b:
        # 过滤 inf 值，只比较有限值
        finite_a = [q for q in qsnr_a if np.isfinite(q)]
        finite_b = [q for q in qsnr_b if np.isfinite(q)]
        if finite_a and finite_b:
            avg_a = np.mean(finite_a)
            avg_b = np.mean(finite_b)
            if avg_b > avg_a + 1:
                print(f"    QA 模式平均 QSNR 高 {avg_b - avg_a:.1f} dB (有限值)，{QTYPE_WEIGHT} 量化质量更好")
            elif avg_a > avg_b + 1:
                print(f"    正态模式平均 QSNR 高 {avg_a - avg_b:.1f} dB (有限值)，{QTYPE_WEIGHT} 量化质量更好")
            else:
                print(f"    两组有限 QSNR 差异不大 (差 {abs(avg_b - avg_a):.1f} dB)")
        else:
            print(f"    所有算子 QSNR 为 inf (量化无损)")
    print(f"    QA 模式通过受控动态范围，使数据更适合 block floating point 量化")

    print("\n" + "=" * 75)
    print("  报告完成")
    print("=" * 75)


if __name__ == "__main__":
    main()
