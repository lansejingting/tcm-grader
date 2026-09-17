"""
Benchmark Runner：纵向对比测试
==============================
- 每个品种抽取 5 张 → 20 × 5 = 100 张
- 完整流水线：品种识别 → 定级 → 评估
- 执行 3 次，纵向对比
- 记录：耗时、token 消耗、品种识别准确率、定级一致性
- 产出：benchmark_report.json（含 3 次完整数据 + 对比汇总）

用法: python 06_benchmark.py
"""

import json
import random
import sys
import time
from pathlib import Path
from collections import defaultdict

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.resolve()))
import config

# 复用 Step3 / Step4 的核心函数
from importlib.machinery import SourceFileLoader

step3 = SourceFileLoader("step3", str(Path(__file__).parent / "03_grade_herbs.py")).load_module()
step4 = SourceFileLoader("step4", str(Path(__file__).parent / "04_evaluate.py")).load_module()


def sample_images(dataset_dir: Path, per_herb: int = 5, seed: int = 42) -> list[dict]:
    """每个品种随机抽取 N 张，返回 [{true_herb, image_path}, ...]。"""
    random.seed(seed)
    results = []
    for herb_dir in sorted(dataset_dir.iterdir()):
        if not herb_dir.is_dir() or herb_dir.name.startswith("."):
            continue
        true_herb = herb_dir.name
        all_imgs = sorted([p for p in herb_dir.iterdir() if p.suffix.lower() in config.IMAGE_EXTENSIONS])
        if not all_imgs:
            continue
        pick = min(per_herb, len(all_imgs))
        chosen = random.sample(all_imgs, pick)
        for p in chosen:
            results.append({"true_herb": true_herb, "image_path": str(p)})
    random.shuffle(results)  # 打乱顺序，避免品种聚类
    print(f"  抽样完成：{len(results)} 张（{len(set(r['true_herb'] for r in results))} 品种 × {per_herb}张）")
    return results


def run_pipeline(retriever, images: list[dict]) -> tuple[list[dict], dict]:
    """
    跑完整流水线，返回 (predictions, token_stats)。
    token_stats = {identify: {...}, grade: {...}, total: {...}}
    """
    token_stats = {
        "identify": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "calls": 0},
        "grade":    {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "calls": 0},
        "total":    {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "calls": 0},
    }

    # Monkey-patch call_vision_api 来捕获 token usage
    original_call = step3.call_vision_api

    def patched_call(system_prompt, user_prompt, image_data_url, stage_key):
        """包装原始调用，捕获 token。stage_key = 'identify' 或 'grade'"""
        if not config.OPENAI_API_KEY:
            raise RuntimeError("OPENAI_API_KEY 未设置")
        from openai import OpenAI
        client = OpenAI(api_key=config.OPENAI_API_KEY, base_url=config.OPENAI_BASE_URL)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": [
                {"type": "text", "text": user_prompt},
                {"type": "image_url", "image_url": {"url": image_data_url}},
            ]},
        ]
        resp = client.chat.completions.create(
            model=config.VISION_MODEL,
            messages=messages,
            temperature=0.1,
            max_tokens=1024,
            response_format={"type": "json_object"},
        )
        # 记录 token
        usage = resp.usage
        ts = token_stats[stage_key]
        ts["prompt_tokens"] += usage.prompt_tokens
        ts["completion_tokens"] += usage.completion_tokens
        ts["total_tokens"] += usage.total_tokens
        ts["calls"] += 1
        token_stats["total"]["prompt_tokens"] += usage.prompt_tokens
        token_stats["total"]["completion_tokens"] += usage.completion_tokens
        token_stats["total"]["total_tokens"] += usage.total_tokens
        token_stats["total"]["calls"] += 1
        return resp.choices[0].message.content

    predictions = []
    t0 = time.time()
    for i, item in enumerate(images, 1):
        true_herb = item["true_herb"]
        image_path = item["image_path"]
        image_data_url = step3.load_and_prepare_image(image_path)

        # 阶段一：品种识别（用 patched_call）
        herb_list_str = "、".join(config.HERB_LIST)
        sys_prompt = step3.HERB_IDENTIFY_PROMPT.format(herb_list=herb_list_str)
        user_prompt = "请从上述 20 种中药材中选择最匹配的品种，输出 JSON。"
        last_err = None
        ident_result = None
        for attempt in range(config.MAX_RETRIES):
            try:
                raw = patched_call(sys_prompt, user_prompt, image_data_url, "identify")
                ident_result = step3.parse_llm_json(raw)
                break
            except Exception as e:
                last_err = e
                if attempt < config.MAX_RETRIES - 1:
                    time.sleep(config.RETRY_DELAY * (attempt + 1))
        if ident_result is None:
            ident_result = {"herb_name": "", "confidence": 0, "reasoning": str(last_err)}

        identified_herb = ident_result.get("herb_name", "")
        if identified_herb and identified_herb not in config.HERB_LIST:
            import difflib
            matches = difflib.get_close_matches(identified_herb, config.HERB_LIST, n=1, cutoff=0.4)
            identified_herb = matches[0] if matches else identified_herb
        herb_ok = identified_herb == true_herb

        # 阶段二：定级（用 patched_call）
        grade_result = {"status": "error", "error": "未执行"}
        hits = retriever.retrieve(identified_herb, query_text=None, top_k=8) if identified_herb else []
        if hits:
            rules_parts = []
            for h in hits:
                crit = h.get("criteria", {})
                vc = h.get("visual_conditions", {})
                crit_text = "；".join(f"{k}: {v}" for k, v in crit.items() if v) or "（未填写）"
                vc_parts = []
                for vk, vv in vc.items():
                    if isinstance(vv, list):
                        vc_parts.append(f"{vk}: {'/'.join(str(x) for x in vv)}")
                    elif vv:
                        vc_parts.append(f"{vk}: {vv}")
                vc_text = "；".join(vc_parts)
                code = h.get("grade_code", "")
                rules_parts.append(f"## {h['grade_tier']}({code})\n定级标准：{crit_text}\n视觉判定：{vc_text}")
            rules_text = "\n\n".join(rules_parts)

            grade_tiers = [h["grade_tier"] for h in hits]
            grade_codes = [h.get("grade_code", "") for h in hits]
            grade_tiers_str = "|".join(grade_tiers)
            grade_code_str = "|".join(grade_codes)
            grade_map_str = "、".join(f"{t}→{c}" for t, c in zip(grade_tiers, grade_codes))
            sys_prompt = step3.GRADING_SYSTEM_PROMPT.format(
                grade_tiers_str=grade_tiers_str, grade_code_str=grade_code_str, grade_map_str=grade_map_str,
            )
            user_prompt = (
                f"【已识别品种】{identified_herb}\n\n"
                f"【该品种定级规则】\n{rules_text}\n\n"
                f"【药材图片】见附件。请按木桶效应逐维度比对，输出 JSON。"
            )
            for attempt in range(config.MAX_RETRIES):
                try:
                    raw = patched_call(sys_prompt, user_prompt, image_data_url, "grade")
                    result_json = step3.parse_llm_json(raw)
                    if not result_json:
                        raise ValueError("JSON 解析失败")
                    conf = float(result_json.get("confidence", 0))
                    if conf < config.REVIEW_CONFIDENCE_THRESHOLD:
                        result_json["review_flag"] = True
                        result_json["review_reason"] = (
                            result_json.get("review_reason", "")
                            + f" 置信度 {conf:.2f} < 阈值 {config.REVIEW_CONFIDENCE_THRESHOLD}"
                        ).strip()
                    grade_result = {
                        "status": "ok", "result": result_json,
                        "retrieved_rules": [{"grade": h["grade_tier"], "grade_code": h.get("grade_code", ""), "score": h.get("score", 1.0)} for h in hits],
                    }
                    break
                except Exception as e:
                    last_err = e
                    if attempt < config.MAX_RETRIES - 1:
                        time.sleep(config.RETRY_DELAY * (attempt + 1))
            if grade_result["status"] != "ok":
                grade_result = {"status": "error", "error": str(last_err)}
        else:
            grade_result = {"status": "error", "error": f"未检索到 {identified_herb} 的定级规则"}

        # 合并
        if grade_result["status"] == "ok":
            r = grade_result["result"]
            if not herb_ok:
                r["review_flag"] = True
                r["review_reason"] = (r.get("review_reason", "") + " 品种识别错误，定级结果仅供参考").strip()
            predictions.append({
                "true_herb": true_herb,
                "identified_herb": identified_herb,
                "herb_confidence": ident_result.get("confidence", 0),
                "herb_identification_correct": herb_ok,
                "image_path": image_path,
                "status": "ok",
                "result": r,
                "retrieved_rules": grade_result.get("retrieved_rules", []),
            })
        else:
            predictions.append({
                "true_herb": true_herb,
                "identified_herb": identified_herb,
                "herb_confidence": ident_result.get("confidence", 0),
                "herb_identification_correct": herb_ok,
                "image_path": image_path,
                "status": "error",
                "error": grade_result.get("error", "定级失败"),
            })

        ok = predictions[-1]["status"] == "ok"
        grade = predictions[-1]["result"].get("grade", "?") if ok else "ERR"
        conf = predictions[-1]["result"].get("confidence", 0) if ok else 0
        print(f"[{i}/{len(images)}] {true_herb} | {Path(image_path).name} → "
              f"{'✅' if herb_ok else '❌'} → {grade} ({conf:.2f})")

        if i % config.BATCH_SIZE == 0:
            time.sleep(0.3)

    elapsed = time.time() - t0
    token_stats["elapsed_seconds"] = round(elapsed, 1)
    token_stats["images_processed"] = len(images)
    token_stats["time_per_image"] = round(elapsed / len(images), 2) if images else 0

    return predictions, token_stats


def run_single_benchmark(images: list[dict], run_index: int, retriever) -> dict:
    """执行单次 benchmark，返回完整结果 dict。"""
    print(f"\n{'='*60}")
    print(f"第 {run_index}/3 次运行")
    print(f"{'='*60}")

    # 1. 流水线
    predictions, token_stats = run_pipeline(retriever, images)

    # 2. 评估
    report = {}
    report.update(step4.evaluate_herb_identification(predictions))
    report.update(step4.evaluate_grading_consistency(predictions))

    # 3. 汇总本次数据
    run_result = {
        "run_index": run_index,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "token_stats": token_stats,
        "herb_identification": report.get("herb_identification", {}),
        "grading_consistency": report.get("grading_consistency", {}),
        "predictions": predictions,  # 保留完整数据供后续分析
    }
    return run_result


def compare_runs(runs: list[dict]) -> dict:
    """纵向对比 3 次运行结果。"""
    comparison = {
        "num_runs": len(runs),
        "runs": [],
        "longitudinal_comparison": {},
    }

    # 提取每次的关键指标
    for r in runs:
        ident = r.get("herb_identification", {})
        cons = r.get("grading_consistency", {})
        toks = r.get("token_stats", {})
        comparison["runs"].append({
            "run_index": r["run_index"],
            "timestamp": r["timestamp"],
            "herb_accuracy": ident.get("accuracy", 0),
            "herb_macro_f1": ident.get("macro_f1", 0),
            "grade_rule_consistency": cons.get("rule_consistency", {}).get("ratio", 0),
            "mean_confidence": cons.get("score_variance", {}).get("all_samples", {}).get("mean", 0),
            "review_trigger_ratio": cons.get("review_flag_analysis", {}).get("trigger_ratio", 0),
            "elapsed_seconds": toks.get("elapsed_seconds", 0),
            "total_tokens": toks.get("total", {}).get("total_tokens", 0),
            "identify_tokens": toks.get("identify", {}).get("total_tokens", 0),
            "grade_tokens": toks.get("grade", {}).get("total_tokens", 0),
        })

    # 纵向对比：均值 / 标准差 / 极差
    metrics = ["herb_accuracy", "herb_macro_f1", "grade_rule_consistency",
               "mean_confidence", "review_trigger_ratio",
               "elapsed_seconds", "total_tokens"]

    for metric in metrics:
        values = [r[metric] for r in comparison["runs"]]
        comparison["longitudinal_comparison"][metric] = {
            "values": values,
            "mean": round(float(np.mean(values)), 4),
            "std": round(float(np.std(values)), 4),
            "min": round(float(np.min(values)), 4),
            "max": round(float(np.max(values)), 4),
            "range": round(float(np.max(values) - np.min(values)), 4),
        }

    # 品种识别一致性对比：同一张图 3 次识别结果是否一致
    # 提取图片路径 → 3 次识别结果
    image_ident = defaultdict(list)
    for run in runs:
        for pred in run.get("predictions", []):
            image_ident[pred["image_path"]].append(pred.get("identified_herb", ""))

    consistent_count = 0
    inconsistent_count = 0
    for img, idents in image_ident.items():
        if len(idents) >= 3:
            if len(set(idents)) == 1:
                consistent_count += 1
            else:
                inconsistent_count += 1

    comparison["identification_stability"] = {
        "images_checked": consistent_count + inconsistent_count,
        "consistent_3x": consistent_count,
        "inconsistent_3x": inconsistent_count,
        "stability_ratio": round(consistent_count / (consistent_count + inconsistent_count), 4) if (consistent_count + inconsistent_count) else 0,
    }

    return comparison


def main():
    print("╔══════════════════════════════════════════════════════════╗")
    print("║          TCM-Grader Benchmark（纵向对比 × 3 次）         ║")
    print("╚══════════════════════════════════════════════════════════╝")

    # 0. 检查 API Key
    if not config.OPENAI_API_KEY:
        print("[ERROR] 请先设置 DASHSCOPE_API_KEY 环境变量")
        return

    # 1. 抽样（固定 seed 保证 3 次用同一张图）
    images = sample_images(config.DATASET_DIR, per_herb=5, seed=42)
    if not images:
        print("[ERROR] 数据集目录下未找到图片")
        return

    # 2. 初始化检索器
    from importlib.machinery import SourceFileLoader
    retriever_mod = SourceFileLoader("retriever", str(Path(__file__).parent / "02_retrieve_rules.py")).load_module()
    retriever = retriever_mod.RuleRetriever()

    # 3. 跑 3 次
    all_runs = []
    for run_idx in range(1, 4):
        run_result = run_single_benchmark(images, run_idx, retriever)
        all_runs.append(run_result)

    # 4. 纵向对比
    print(f"\n{'='*60}")
    print("纵向对比汇总")
    print(f"{'='*60}")
    comparison = compare_runs(all_runs)

    # 5. 输出完整报告
    full_report = {
        "metadata": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "dataset": str(config.DATASET_DIR),
            "images_per_herb": 5,
            "total_images": len(images),
            "num_runs": 3,
            "model": config.VISION_MODEL,
            "embedding_model": config.EMBEDDING_MODEL,
        },
        "runs": all_runs,
        "comparison": comparison,
    }

    report_path = config.RESULTS_DIR / "benchmark_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(full_report, f, ensure_ascii=False, indent=2, default=str)

    # 打印对比摘要
    comp = comparison["longitudinal_comparison"]
    print(f"\n{'指标':<25s} {'Run1':>10s} {'Run2':>10s} {'Run3':>10s} {'均值':>10s} {'标准差':>10s}")
    print("-" * 75)
    for metric, info in comp.items():
        vals = info["values"]
        print(f"{metric:<25s} {vals[0]:>10.4f} {vals[1]:>10.4f} {vals[2]:>10.4f} {info['mean']:>10.4f} {info['std']:>10.4f}")

    stab = comparison["identification_stability"]
    print(f"\n品种识别稳定性（同图 3 次结果一致性）：")
    print(f"  一致：{stab['consistent_3x']}/{stab['images_checked']} = {stab['stability_ratio']:.2%}")

    # Token 汇总
    print(f"\nToken 消耗汇总：")
    total_ident = sum(r["token_stats"]["identify"]["total_tokens"] for r in all_runs)
    total_grade = sum(r["token_stats"]["grade"]["total_tokens"] for r in all_runs)
    total_all = sum(r["token_stats"]["total"]["total_tokens"] for r in all_runs)
    print(f"  品种识别：{total_ident:,} tokens")
    print(f"  定级推理：{total_grade:,} tokens")
    print(f"  合计：{total_all:,} tokens（{total_all/3/len(images):.0f} tokens/图/次）")

    print(f"\n完整报告：{report_path}")


if __name__ == "__main__":
    main()
