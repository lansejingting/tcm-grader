"""
易错品种识别测试（优化 Prompt 后 vs benchmark 旧结果对比）
==========================================================
- 复用 06_benchmark.py 的固定抽样（seed=42，同一批图片，可公平对比）
- 只跑阶段一：品种识别（带形态特征表 + 混淆对区分要点的新 Prompt）
- 与 benchmark_report.json 里旧 Prompt 的识别结果逐品种对比

用法: python 09_test_error_prone.py
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.resolve()))
import config
from importlib.machinery import SourceFileLoader

step3 = SourceFileLoader("step3", str(Path(__file__).parent / "03_grade_herbs.py")).load_module()
bench = SourceFileLoader("bench", str(Path(__file__).parent / "06_benchmark.py")).load_module()

# benchmark 顽固错误 / 高频混淆品种
ERROR_PRONE_HERBS = [
    "金樱子", "覆盆子", "桃仁", "地肤子", "砂仁", "川楝子",
    "连翘", "木瓜", "山茱萸", "豆蔻", "瓜蒌", "五味子",
]

BENCHMARK_REPORT = Path(__file__).parent / "results" / "benchmark_report.json"


def load_old_results() -> dict:
    """从 benchmark_report.json 提取旧 Prompt 的逐图识别结果：{image_path: (identified, correct)}"""
    with open(BENCHMARK_REPORT, "r", encoding="utf-8") as f:
        r = json.load(f)
    old = {}
    for pred in r["runs"][0]["predictions"]:
        old[pred["image_path"]] = (pred.get("identified_herb", ""), pred.get("herb_identification_correct", False))
    return old


def main():
    if not config.OPENAI_API_KEY:
        print("[ERROR] 请先设置 DASHSCOPE_API_KEY")
        return

    # 1. 固定抽样（与 benchmark 完全同一批图）
    all_images = bench.sample_images(config.DATASET_DIR, per_herb=5, seed=42)
    images = [x for x in all_images if x["true_herb"] in ERROR_PRONE_HERBS]
    print(f"易错品种测试：{len(images)} 张（{len(set(x['true_herb'] for x in images))} 品种 × 5 张）\n")

    # 2. 旧结果
    old_results = load_old_results()

    # 3. 用新 Prompt 逐张识别（只跑阶段一）
    from openai import OpenAI
    client = OpenAI(api_key=config.OPENAI_API_KEY, base_url=config.OPENAI_BASE_URL)

    new_results = {}
    t0 = time.time()
    for i, item in enumerate(images, 1):
        img_path, true_herb = item["image_path"], item["true_herb"]
        image_data_url = step3.load_and_prepare_image(img_path)
        try:
            ident = step3.identify_herb(image_data_url)
            identified = ident["identified_herb"]
            conf = ident["herb_confidence"]
            ok = identified == true_herb
        except Exception as e:
            identified, conf, ok = f"ERR:{e}", 0, False

        new_results[img_path] = (identified, ok, conf)
        old_ident, old_ok = old_results.get(img_path, ("?", False))
        old_mark = "✅" if old_ok else "❌"
        new_mark = "✅" if ok else "❌"
        change = "" if old_ok == ok else ("  ↑修复" if ok else "  ↓退化")
        print(f"[{i}/{len(images)}] {true_herb:<6s} | 旧{old_mark}{old_ident:<6s} → 新{new_mark}{identified:<6s} conf={conf:.2f}{change}")

        with open(Path(__file__).parent / "results" / "error_prone_test.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps({
                "true_herb": true_herb, "image_path": img_path,
                "identified": identified, "correct": ok, "confidence": conf,
            }, ensure_ascii=False) + "\n")

    # 4. 汇总对比
    print(f"\n{'='*66}")
    print(f"{'品种':<8s} {'旧Prompt正确':>12s} {'新Prompt正确':>12s} {'变化':>8s}")
    print("-" * 66)
    total_old = total_new = 0
    for herb in ERROR_PRONE_HERBS:
        herb_imgs = [x["image_path"] for x in images if x["true_herb"] == herb]
        old_c = sum(1 for p in herb_imgs if old_results.get(p, ("", False))[1])
        new_c = sum(1 for p in herb_imgs if new_results.get(p, ("", False, 0))[1])
        total_old += old_c
        total_new += new_c
        diff = new_c - old_c
        mark = "" if diff == 0 else (f"{'+' if diff > 0 else ''}{diff}")
        print(f"{herb:<8s} {old_c:>8d}/5   {new_c:>8d}/5   {mark:>6s}")
    print("-" * 66)
    print(f"{'合计':<8s} {total_old:>8d}/{len(images)}  {total_new:>8d}/{len(images)}  "
          f"{'+' if total_new - total_old >= 0 else ''}{total_new - total_old}")
    print(f"\n新Prompt识别准确率: {total_new / len(images):.1%}（旧: {total_old / len(images):.1%}）")
    print(f"耗时: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
