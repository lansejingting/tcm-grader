"""
Demo：展示单张金樱子图片的完整定级过程（含所有中间产物）
========================================================
阶段 0: 图片加载与预处理
阶段 1: 品种识别（20 选 1）—— 展示 Prompt + 模型返回
阶段 2: 规则检索 —— 展示检索到的定级规则
阶段 3: 定级推理 —— 展示定级 Prompt + 模型原始返回
阶段 4: 结果解析 —— 展示最终 JSON + 置信度/复核判定

用法: python 08_demo_process.py [--image 路径] [--herb 真实品种名]
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.resolve()))
import config
from importlib.machinery import SourceFileLoader

step3 = SourceFileLoader("step3", str(Path(__file__).parent / "03_grade_herbs.py")).load_module()


def print_stage(title: str):
    print(f"\n{'█' * 66}")
    print(f"█ {title}")
    print(f"{'█' * 66}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default=str(config.DATASET_DIR / "金樱子" / "Rosae_Laevigatae_Frucyus0.jpg"))
    ap.add_argument("--herb", default="金樱子")
    args = ap.parse_args()

    # ========== 阶段 0: 图片加载 ==========
    print_stage("阶段 0：图片加载与预处理")
    print(f"原始路径: {args.image}")
    p = Path(args.image)
    print(f"文件存在: {p.exists()}  大小: {p.stat().st_size / 1024:.0f} KB")

    from PIL import Image
    img = Image.open(p)
    print(f"原始尺寸: {img.size[0]}×{img.size[1]}  格式: {img.format}  模式: {img.mode}")

    image_data_url = step3.load_and_prepare_image(args.image)
    b64_len = len(image_data_url)
    print(f"预处理后: 长边≤{config.MAX_IMAGE_SIDE}px → JPEG quality=90 → base64 {b64_len // 1024} KB")
    print(f"data URL 前 60 字符: {image_data_url[:60]}...")

    # ========== 阶段 1: 品种识别 ==========
    print_stage("阶段 1：品种识别（20 选 1，不告诉模型真实品种）")
    herb_list_str = "、".join(config.HERB_LIST)
    sys_prompt = step3.HERB_IDENTIFY_PROMPT.format(herb_list=herb_list_str)

    print("--- 发给模型的 System Prompt ---")
    print(sys_prompt)
    print("\n--- User Prompt ---")
    print("请从上述 20 种中药材中选择最匹配的品种，输出 JSON。 + [图片附件]")

    from openai import OpenAI
    client = OpenAI(api_key=config.OPENAI_API_KEY, base_url=config.OPENAI_BASE_URL)

    resp = client.chat.completions.create(
        model=config.VISION_MODEL,
        messages=[
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": [
                {"type": "text", "text": "请从上述 20 种中药材中选择最匹配的品种，输出 JSON。"},
                {"type": "image_url", "image_url": {"url": image_data_url}},
            ]},
        ],
        temperature=0.1, max_tokens=1024,
        response_format={"type": "json_object"},
    )
    raw_ident = resp.choices[0].message.content
    print(f"\n--- 模型原始返回（tokens: prompt={resp.usage.prompt_tokens}, completion={resp.usage.completion_tokens}）---")
    print(raw_ident)

    ident = step3.parse_llm_json(raw_ident)
    identified_herb = ident.get("herb_name", "")
    herb_ok = identified_herb == args.herb
    print(f"\n>>> 识别结果: {identified_herb}  置信度: {ident.get('confidence')}  "
          f"真实: {args.herb}  {'✅ 正确' if herb_ok else '❌ 错误'}")
    print(f">>> 识别依据: {ident.get('reasoning', '')}")

    # ========== 阶段 2: 规则检索 ==========
    print_stage("阶段 2：RAG 规则检索（品种已定 → 精确过滤取全部等级）")
    retriever_mod = SourceFileLoader("retriever", str(Path(__file__).parent / "02_retrieve_rules.py")).load_module()
    retriever = retriever_mod.RuleRetriever()
    hits = retriever.retrieve(identified_herb, query_text=None, top_k=8)

    print(f"检索方式: herb_name='{identified_herb}' 精确过滤（未用语义检索）")
    print(f"命中 {len(hits)} 条规则:")
    for h in hits:
        print(f"  - {h['grade_tier']}({h.get('grade_code', '')})  score={h.get('score', 1.0)}")

    # ========== 阶段 3: 定级推理 ==========
    print_stage("阶段 3：定级推理（木桶效应 + 一票否决）")
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
    grade_map_str = "、".join(f"{t}→{c}" for t, c in zip(grade_tiers, grade_codes))
    grade_sys = step3.GRADING_SYSTEM_PROMPT.format(
        grade_tiers_str="|".join(grade_tiers),
        grade_code_str="|".join(grade_codes),
        grade_map_str=grade_map_str,
    )
    grade_user = (
        f"【已识别品种】{identified_herb}\n\n"
        f"【该品种定级规则】\n{rules_text}\n\n"
        f"【药材图片】见附件。请按木桶效应逐维度比对，输出 JSON。"
    )

    print("--- 发给模型的 User Prompt（System 见 03_grade_herbs.py 的 GRADING_SYSTEM_PROMPT）---")
    print(grade_user)

    resp2 = client.chat.completions.create(
        model=config.VISION_MODEL,
        messages=[
            {"role": "system", "content": grade_sys},
            {"role": "user", "content": [
                {"type": "text", "text": grade_user},
                {"type": "image_url", "image_url": {"url": image_data_url}},
            ]},
        ],
        temperature=0.1, max_tokens=1024,
        response_format={"type": "json_object"},
    )
    raw_grade = resp2.choices[0].message.content
    print(f"\n--- 模型原始返回（tokens: prompt={resp2.usage.prompt_tokens}, completion={resp2.usage.completion_tokens}）---")
    print(raw_grade)

    # ========== 阶段 4: 结果解析 ==========
    print_stage("阶段 4：结果解析与复核判定")
    result = step3.parse_llm_json(raw_grade)
    conf = float(result.get("confidence", 0))
    review = False
    review_reasons = []
    if conf < config.REVIEW_CONFIDENCE_THRESHOLD:
        review = True
        review_reasons.append(f"置信度 {conf:.2f} < 阈值 {config.REVIEW_CONFIDENCE_THRESHOLD}")
    if not herb_ok:
        review = True
        review_reasons.append("品种识别错误，定级结果仅供参考")

    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"\n>>> 最终等级: {result.get('grade')}（{result.get('grade_code')}）  置信度: {conf}")
    print(f">>> 木桶短板维度: {result.get('lowest_dimension')}")
    print(f">>> 人工复核: {'⚠ 触发 → ' + '；'.join(review_reasons) if review else '✅ 不需要'}")


if __name__ == "__main__":
    main()
