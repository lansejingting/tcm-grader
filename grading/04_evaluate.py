"""
Step 4: 评估脚本（双阶段）
--------------------------
A. 品种识别评估（客观指标）
   - Accuracy / Precision / Recall / F1（macro & weighted）
   - Confusion Matrix（品种 × 品种）
   - 按品种统计识别错误分布

B. 定级一致性评估（无人工标签，基于规则自检验）
   - Grade Consistency：品种识别正确的样本中，定级分布合理性
   - Score Variance：置信度分布（均值/标准差/分位）
   - Rule Consistency：定级是否符合该品种已检索规则（等级 code 对齐检查）
   - Review Flag 分析：复核触发比例及原因分布

C. 标签对比（可选，需 --labels）
   若提供人工定级标签，则计算 Accuracy / Kappa / 混淆矩阵

输出：results/evaluation_report.json
"""

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.resolve()))
import config


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


# ============================================================
# A. 品种识别评估
# ============================================================

def evaluate_herb_identification(predictions: list[dict]) -> dict:
    """
    品种识别指标：accuracy / precision / recall / F1 / 混淆矩阵。
    ground truth = true_herb（文件夹名），prediction = identified_herb（模型识别）。
    """
    ok_rows = [r for r in predictions if r.get("status") == "ok"]
    if not ok_rows:
        return {"note": "无有效预测结果"}

    # 收集所有出现过的品种（保证混淆矩阵完整）
    all_herbs = sorted(set(r["true_herb"] for r in ok_rows) | set(r.get("identified_herb", "") for r in ok_rows if r.get("identified_herb")))
    herb_to_idx = {h: i for i, h in enumerate(all_herbs)}
    n = len(all_herbs)

    # 混淆矩阵
    cm = np.zeros((n, n), dtype=int)
    for r in ok_rows:
        true_idx = herb_to_idx.get(r["true_herb"], -1)
        pred_idx = herb_to_idx.get(r.get("identified_herb", ""), -1)
        if true_idx >= 0 and pred_idx >= 0:
            cm[true_idx, pred_idx] += 1

    total = cm.sum()
    # Accuracy
    correct = np.trace(cm)
    accuracy = correct / total if total > 0 else 0

    # 每类 Precision / Recall / F1
    precisions, recalls, f1s = [], [], []
    for i in range(n):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        precisions.append(prec)
        recalls.append(rec)
        f1s.append(f1)

    macro_f1 = np.mean(f1s) if f1s else 0
    weighted_f1 = np.average(f1s, weights=cm.sum(axis=1)) if total > 0 else 0

    # Cohen's Kappa
    po = correct / total if total > 0 else 0
    pe_sum = 0
    for i in range(n):
        pe_sum += (cm[i, :].sum() / total) * (cm[:, i].sum() / total)
    kappa = (po - pe_sum) / (1 - pe_sum) if (1 - pe_sum) != 0 else 0

    # 混淆矩阵格式化（只显示 Top-10 高频品种对）
    cm_formatted = []
    for i, true_h in enumerate(all_herbs):
        row = {"true_herb": true_h}
        for j, pred_h in enumerate(all_herbs):
            row[pred_h] = int(cm[i, j])
        cm_formatted.append(row)

    # 识别错误归因：哪些品种最常被错认成什么
    error_pairs = []
    for i in range(n):
        for j in range(n):
            if i != j and cm[i, j] > 0:
                error_pairs.append((int(cm[i, j]), all_herbs[i], all_herbs[j]))
    error_pairs.sort(reverse=True)

    # 每品种识别统计
    per_herb = {}
    for i, herb in enumerate(all_herbs):
        tp = cm[i, i]
        total_herb = cm[i, :].sum()
        per_herb[herb] = {
            "support": int(total_herb),
            "correct": int(tp),
            "accuracy": round(tp / total_herb, 4) if total_herb > 0 else 0,
            "precision": round(precisions[i], 4),
            "recall": round(recalls[i], 4),
            "f1": round(f1s[i], 4),
        }

    return {
        "herb_identification": {
            "total_samples": int(total),
            "accuracy": round(float(accuracy), 4),
            "macro_f1": round(float(macro_f1), 4),
            "weighted_f1": round(float(weighted_f1), 4),
            "kappa": round(float(kappa), 4),
            "metrics_per_herb": per_herb,
            "confusion_matrix": cm_formatted,
            "top_confusion_pairs": [
                {"true": t, "pred": p, "count": c} for c, t, p in error_pairs[:10]
            ],
        }
    }


# ============================================================
# B. 定级一致性评估
# ============================================================

def evaluate_grading_consistency(predictions: list[dict]) -> dict:
    """
    定级一致性：
    1. 等级分布（整体 + 按品种识别正确/错误分组）
    2. Score Variance：置信度分布
    3. Rule Consistency：定级是否符合已检索规则的等级集合
    4. Review Flag 分析
    5. Grade Consistency（仅品种识别正确的子集）：按品种统计定级分布合理性
    """
    ok_rows = [r for r in predictions if r.get("status") == "ok"]
    if not ok_rows:
        return {"note": "无有效预测结果"}

    # 分级：品种识别正确 vs 错误
    correct_ident = [r for r in ok_rows if r.get("herb_identification_correct")]
    wrong_ident = [r for r in ok_rows if not r.get("herb_identification_correct")]

    # 1. 等级分布
    all_grades = [r["result"].get("grade", "") for r in ok_rows]
    correct_grades = [r["result"].get("grade", "") for r in correct_ident]
    wrong_grades = [r["result"].get("grade", "") for r in wrong_ident]

    grade_dist = {}
    for label, grades in [("all", all_grades), ("correct_ident", correct_grades), ("wrong_ident", wrong_grades)]:
        cnt = Counter(grades)
        total = len(grades)
        grade_dist[label] = {
            g: {"count": cnt.get(g, 0), "ratio": round(cnt.get(g, 0) / total, 3) if total else 0}
            for g in config.GRADE_TIERS
        }

    # 2. Score Variance（置信度分布）
    all_confs = [float(r["result"].get("confidence", 0)) for r in ok_rows]
    correct_confs = [float(r["result"].get("confidence", 0)) for r in correct_ident]
    wrong_confs = [float(r["result"].get("confidence", 0)) for r in wrong_ident]

    def conf_stats(confs):
        if not confs:
            return {"mean": 0, "std": 0, "min": 0, "max": 0, "p25": 0, "p50": 0, "p75": 0}
        arr = np.array(confs)
        return {
            "mean": round(float(np.mean(arr)), 4),
            "std": round(float(np.std(arr)), 4),
            "min": round(float(np.min(arr)), 4),
            "max": round(float(np.max(arr)), 4),
            "p25": round(float(np.percentile(arr, 25)), 4),
            "p50": round(float(np.percentile(arr, 50)), 4),
            "p75": round(float(np.percentile(arr, 75)), 4),
        }

    score_variance = {
        "all_samples": conf_stats(all_confs),
        "herb_ident_correct": conf_stats(correct_confs),
        "herb_ident_wrong": conf_stats(wrong_confs),
    }

    # 3. Rule Consistency：定级结果是否在该品种已检索规则的等级集合内
    rule_consistent = 0
    rule_inconsistent = 0
    inconsistent_examples = []
    for r in ok_rows:
        retrieved_rules = r.get("retrieved_rules", []) or []
        retrieved_grade_tiers = {rr.get("grade", "") for rr in retrieved_rules if isinstance(rr, dict)}
        pred_grade = r["result"].get("grade", "")
        if pred_grade in retrieved_grade_tiers:
            rule_consistent += 1
        else:
            rule_inconsistent += 1
            if len(inconsistent_examples) < 5:
                inconsistent_examples.append({
                    "herb": r.get("identified_herb", ""),
                    "predicted_grade": pred_grade,
                    "retrieved_grades": sorted(retrieved_grade_tiers),
                    "image": Path(r["image_path"]).name,
                })

    # 4. Review Flag 分析
    review_count = sum(1 for r in ok_rows if r["result"].get("review_flag"))
    review_reasons = []
    for r in ok_rows:
        reason = r["result"].get("review_reason", "")
        if reason:
            # 归类：品种识别错误 / 置信度低 / 其他
            if "品种识别错误" in reason:
                review_reasons.append("herb_identification_wrong")
            elif "置信度" in reason:
                review_reasons.append("low_confidence")
            else:
                review_reasons.append("other")
    reason_dist = Counter(review_reasons)

    # 5. Grade Consistency（品种识别正确子集）：按品种统计定级分布
    per_herb_grade_dist = {}
    for r in correct_ident:
        herb = r["identified_herb"]
        g = r["result"].get("grade", "")
        if herb not in per_herb_grade_dist:
            per_herb_grade_dist[herb] = []
        per_herb_grade_dist[herb].append(g)

    per_herb_dist = {}
    for herb, grades in per_herb_grade_dist.items():
        cnt = Counter(grades)
        total = len(grades)
        per_herb_dist[herb] = {
            "total": total,
            "distribution": {
                g: {"count": cnt.get(g, 0), "ratio": round(cnt.get(g, 0) / total, 3)}
                for g in config.GRADE_TIERS
            },
        }

    return {
        "grading_consistency": {
            "total_samples": len(ok_rows),
            "herb_ident_correct_count": len(correct_ident),
            "herb_ident_wrong_count": len(wrong_ident),
            "grade_distribution": grade_dist,
            "score_variance": score_variance,
            "rule_consistency": {
                "consistent": rule_consistent,
                "inconsistent": rule_inconsistent,
                "ratio": round(rule_consistent / (rule_consistent + rule_inconsistent), 4) if (rule_consistent + rule_inconsistent) else 0,
                "inconsistent_examples": inconsistent_examples,
            },
            "review_flag_analysis": {
                "triggered": review_count,
                "trigger_ratio": round(review_count / len(ok_rows), 4) if ok_rows else 0,
                "reason_distribution": dict(reason_dist),
            },
            "per_herb_grade_distribution": per_herb_dist,
        }
    }


# ============================================================
# C. 标签对比（可选，需人工定级标签）
# ============================================================

def evaluate_against_labels(predictions: list[dict], labels: list[dict]) -> dict:
    """
    与人工定级标签对比。labels.jsonl 格式：
    {"image_path": "...", "true_grade": "特级", "true_herb": "乌梅"}
    """
    label_map = {l["image_path"]: l for l in labels}
    aligned = []
    for p in predictions:
        if p.get("status") != "ok":
            continue
        label = label_map.get(p["image_path"])
        if not label:
            continue
        pred_grade = p["result"].get("grade", "")
        true_grade = label.get("true_grade", "")
        if true_grade in config.GRADE_TIERS and pred_grade in config.GRADE_TIERS:
            aligned.append((pred_grade, true_grade, p.get("identified_herb", ""), p.get("true_herb", "")))

    if not aligned:
        return {"note": "无对齐的预测-标签对"}

    all_tiers = list(config.GRADE_TIERS)
    tier_to_idx = {t: i for i, t in enumerate(all_tiers)}
    y_true = [tier_to_idx[t] for _, t, _, _ in aligned]
    y_pred = [tier_to_idx[p] for p, _, _, _ in aligned]

    n = len(all_tiers)
    cm = np.zeros((n, n), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1

    precisions, recalls, f1s = [], [], []
    for i in range(n):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        precisions.append(prec)
        recalls.append(rec)
        f1s.append(f1)

    total = len(y_true)
    po = np.mean(np.array(y_true) == np.array(y_pred))
    pe_sum = sum((cm[i, :].sum() / total) * (cm[:, i].sum() / total) for i in range(n))
    kappa = (po - pe_sum) / (1 - pe_sum) if (1 - pe_sum) != 0 else 0.0

    # 分级：品种识别正确 vs 错误的定级准确率
    correct_ident_grades = [(p, t) for p, t, ident, true in aligned if ident == true]
    wrong_ident_grades = [(p, t) for p, t, ident, true in aligned if ident != true]

    def simple_acc(pairs):
        return sum(1 for p, t in pairs if p == t) / len(pairs) if pairs else 0

    cm_formatted = []
    for i, true_tier in enumerate(all_tiers):
        row = {"true": true_tier}
        for j, pred_tier in enumerate(all_tiers):
            row[pred_tier] = int(cm[i, j])
        cm_formatted.append(row)

    return {
        "ground_truth_comparison": {
            "aligned_samples": len(aligned),
            "accuracy": round(float(po), 4),
            "kappa": round(float(kappa), 4),
            "accuracy_when_herb_correct": round(simple_acc(correct_ident_grades), 4),
            "accuracy_when_herb_wrong": round(simple_acc(wrong_ident_grades), 4),
            "metrics_per_grade": {
                all_tiers[i]: {
                    "precision": round(precisions[i], 4),
                    "recall": round(recalls[i], 4),
                    "f1": round(f1s[i], 4),
                    "support": int(cm[i, :].sum()),
                }
                for i in range(n)
            },
            "macro_f1": round(float(np.mean(f1s)), 4),
            "weighted_f1": round(float(np.average(f1s, weights=cm.sum(axis=1))), 4),
            "confusion_matrix": cm_formatted,
        }
    }


def main():
    parser = argparse.ArgumentParser(description="Step4: 双阶段评估（品种识别 + 定级一致性）")
    parser.add_argument("--predictions", type=Path,
                        default=config.RESULTS_DIR / "predictions.jsonl")
    parser.add_argument("--labels", type=Path, default=None,
                        help="可选：人工定级标签（JSONL，含 image_path/true_grade）")
    parser.add_argument("--output", type=Path,
                        default=config.RESULTS_DIR / "evaluation_report.json")
    args = parser.parse_args()

    if not args.predictions.exists():
        print(f"[ERROR] 预测文件不存在：{args.predictions}")
        print("请先运行 Step3：python 03_grade_herbs.py")
        return

    predictions = load_jsonl(args.predictions)
    print(f"[1/4] 加载预测结果：{len(predictions)} 条")

    # A. 品种识别评估
    print("[2/4] 品种识别评估（客观指标）...")
    report_ident = evaluate_herb_identification(predictions)

    # B. 定级一致性评估
    print("[3/4] 定级一致性评估（规则自检验）...")
    report_grade = evaluate_grading_consistency(predictions)

    report = {}
    report.update(report_ident)
    report.update(report_grade)

    # C. 标签对比
    if args.labels and args.labels.exists():
        print(f"[4/4] 标签对比评估：{args.labels}")
        labels = load_jsonl(args.labels)
        report.update(evaluate_against_labels(predictions, labels))
    else:
        print("[4/4] 未提供 --labels，跳过标签对比。")

    # 写入报告
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    # 打印摘要
    print(f"\n{'='*60}")
    print("评估摘要")
    print(f"{'='*60}")

    ident = report.get("herb_identification", {})
    if ident:
        print(f"\n【品种识别】")
        print(f"  Accuracy:      {ident.get('accuracy', '?')}")
        print(f"  Macro F1:      {ident.get('macro_f1', '?')}")
        print(f"  Weighted F1:   {ident.get('weighted_f1', '?')}")
        print(f"  Kappa:         {ident.get('kappa', '?')}")
        top_conf = ident.get("top_confusion_pairs", [])
        if top_conf:
            print(f"  最易混淆:      {top_conf[0]['true']} ↔ {top_conf[0]['pred']} ({top_conf[0]['count']}次)")

    cons = report.get("grading_consistency", {})
    if cons:
        print(f"\n【定级一致性】")
        print(f"  规则一致性:     {cons.get('rule_consistency', {}).get('ratio', '?')}")
        print(f"  置信度均值:     {cons.get('score_variance', {}).get('all_samples', {}).get('mean', '?')}")
        print(f"  复核触发率:     {cons.get('review_flag_analysis', {}).get('trigger_ratio', '?')}")
        print(f"  识别正确的定级样本: {cons.get('herb_ident_correct_count', '?')}/{cons.get('total_samples', '?')}")

    gt = report.get("ground_truth_comparison")
    if gt:
        print(f"\n【标签对比】")
        print(f"  Accuracy:      {gt.get('accuracy', '?')}")
        print(f"  Kappa:         {gt.get('kappa', '?')}")
        print(f"  识别正确时定级准确率: {gt.get('accuracy_when_herb_correct', '?')}")
        print(f"  识别错误时定级准确率: {gt.get('accuracy_when_herb_wrong', '?')}")

    print(f"\n完整报告：{args.output}")


if __name__ == "__main__":
    main()
