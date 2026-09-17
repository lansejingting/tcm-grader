"""
Benchmark 报告解析器
====================
解析 results/benchmark_report.json，打印人可读的摘要。

用法:
    python 07_parse_benchmark.py            # 打印完整摘要
    python 07_parse_benchmark.py --run 2    # 只看第 2 次运行的详细数据
    python 07_parse_benchmark.py --bad      # 列出 3 次都识别错的顽固 bad case
"""

import argparse
import json
from collections import Counter
from pathlib import Path

REPORT = Path(__file__).parent / "results" / "benchmark_report.json"


def print_overview(r):
    m = r["metadata"]
    print("=" * 70)
    print("一、测试概况")
    print("=" * 70)
    print(f"  时间: {m['timestamp']}  模型: {m['model']}")
    print(f"  数据: {m['images_per_herb']} 品种/张 × {m['total_images']} 张 × {m['num_runs']} 次")


def print_runs_table(r):
    print()
    print("=" * 70)
    print("二、三次运行横向对比")
    print("=" * 70)
    runs = r["comparison"]["runs"]
    header = f"  {'指标':<24s}" + "".join(f"{'Run' + str(x['run_index']):>10s}" for x in runs) + f"{'均值':>10s}{'标准差':>10s}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    comp = r["comparison"]["longitudinal_comparison"]
    label_map = [
        ("herb_accuracy", "品种识别 Accuracy"),
        ("herb_macro_f1", "品种识别 Macro F1"),
        ("grade_rule_consistency", "规则一致性"),
        ("mean_confidence", "定级置信度均值"),
        ("review_trigger_ratio", "复核触发率"),
        ("elapsed_seconds", "耗时(秒)"),
        ("total_tokens", "总 Token"),
    ]
    for key, label in label_map:
        info = comp[key]
        vals = info["values"]
        row = f"  {label:<24s}" + "".join(f"{v:>10.4f}" if v < 1000 else f"{v:>10,.0f}" for v in vals)
        row += f"{info['mean']:>10.4f}" if info["mean"] < 1000 else f"{info['mean']:>10,.0f}"
        row += f"{info['std']:>10.4f}" if info["std"] < 1000 else f"{info['std']:>10,.0f}"
        print(row)


def print_stability(r):
    stab = r["comparison"]["identification_stability"]
    print()
    print("=" * 70)
    print("三、识别稳定性（同一张图 3 次识别结果是否一致）")
    print("=" * 70)
    print(f"  3 次全一致: {stab['consistent_3x']}/{stab['images_checked']} = {stab['stability_ratio']:.2%}")
    print(f"  存在摇摆:   {stab['inconsistent_3x']} 张 ← 这些图片是模型决策边界，值得人工复核")


def print_confusion(r, run_idx=None):
    print()
    print("=" * 70)
    print("四、最易混淆品种对" + (f"（Run {run_idx}）" if run_idx else "（Run 1）"))
    print("=" * 70)
    run = r["runs"][(run_idx or 1) - 1]
    pairs = run["herb_identification"].get("top_confusion_pairs", [])
    if not pairs:
        print("  无混淆数据")
        return
    for p in pairs:
        print(f"  真实 {p['true']:<8s} → 误判为 {p['pred']:<8s} × {p['count']} 次")


def print_token_detail(r):
    print()
    print("=" * 70)
    print("五、Token 消耗明细")
    print("=" * 70)
    n_img = r["metadata"]["total_images"]
    tot_i = tot_g = 0
    for run in r["runs"]:
        ts = run["token_stats"]
        i_tok = ts["identify"]["total_tokens"]
        g_tok = ts["grade"]["total_tokens"]
        tot_i += i_tok
        tot_g += g_tok
        print(f"  Run {run['run_index']}: 识别 {i_tok:>8,} + 定级 {g_tok:>9,} = {i_tok + g_tok:>9,} tokens"
              f"  ({run['token_stats']['time_per_image']}s/张)")
    total = tot_i + tot_g
    print(f"  {'合计':<8s} 识别 {tot_i:>8,} + 定级 {tot_g:>9,} = {total:>9,} tokens")
    print(f"  单张单次平均: {total / (n_img * 3):.0f} tokens  |  识别/定级占比: {tot_i / total:.0%} / {tot_g / total:.0%}")


def print_run_detail(r, run_idx):
    run = r["runs"][run_idx - 1]
    preds = run["predictions"]
    print()
    print("=" * 70)
    print(f"附：Run {run_idx} 每品种识别统计")
    print("=" * 70)
    per = {}
    for p in preds:
        t = p["true_herb"]
        if t not in per:
            per[t] = {"total": 0, "correct": 0}
        per[t]["total"] += 1
        if p.get("herb_identification_correct"):
            per[t]["correct"] += 1
    rows = sorted(per.items(), key=lambda kv: kv[1]["correct"] / kv[1]["total"])
    for herb, s in rows:
        acc = s["correct"] / s["total"]
        bar = "█" * int(acc * 20)
        mark = "✅" if acc == 1 else ("⚠ " if acc >= 0.6 else "❌")
        print(f"  {mark} {herb:<10s} {s['correct']}/{s['total']} = {acc:.0%} {bar}")


def print_stubborn_errors(r):
    print()
    print("=" * 70)
    print("附：顽固 Bad Case（3 次都识别错的图片）")
    print("=" * 70)
    by_img = {}
    for run in r["runs"]:
        for p in run["predictions"]:
            img = p["image_path"]
            if img not in by_img:
                by_img[img] = {"true": p["true_herb"], "preds": []}
            by_img[img]["preds"].append(p.get("identified_herb", ""))

    stubborn = [(img, d) for img, d in by_img.items()
                if len(d["preds"]) == 3 and len(set(d["preds"])) == 1 and d["preds"][0] != d["true"]]
    stubborn.sort(key=lambda x: x[1]["true"])
    if not stubborn:
        print("  无：不存在 3 次都稳定识别错的图片")
        return
    for img, d in stubborn:
        print(f"  {d['true']:<8s} 稳定误判为 {d['preds'][0]:<8s} | {Path(img).name}")


def main():
    ap = argparse.ArgumentParser(description="解析 benchmark_report.json")
    ap.add_argument("--run", type=int, choices=[1, 2, 3], help="查看某次运行的每品种明细")
    ap.add_argument("--bad", action="store_true", help="列出 3 次都识别错的顽固 bad case")
    args = ap.parse_args()

    with open(REPORT, "r", encoding="utf-8") as f:
        r = json.load(f)

    print_overview(r)
    print_runs_table(r)
    print_stability(r)
    print_confusion(r, args.run)
    print_token_detail(r)
    if args.run:
        print_run_detail(r, args.run)
    if args.bad:
        print_stubborn_errors(r)


if __name__ == "__main__":
    main()
