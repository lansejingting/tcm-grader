"""
Step 3: AI 品种识别 + 定级推理（两阶段）
----------------------------------------
遍历 NB-TCM-CHM 数据集目录下的图片：
    阶段一：品种识别（不传文件夹名，纯看图识别）
    阶段二：定级推理（用识别出的品种检索规则，再定级）

文件夹名作为 ground truth label 保留在输出里，供 Step4 评估用。

输出：results/predictions.jsonl
"""

import argparse
import base64
import json
import sys
import time
from io import BytesIO
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.resolve()))
import config

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False


# ============ Prompt 模板 ============

# 易混淆品种对的区分要点（基于 benchmark 高频误判统计整理）
CONFUSION_TIPS = """- **金樱子 vs 覆盆子**（最易混）：金樱子是单一果实、倒卵形、表面红黄色/棕红色且有刺状突起（硬刺）；覆盆子是聚合果、圆锥形、灰绿色/淡棕色、密被灰白色绒毛、小核果易脱落。看到绒毛浓密→覆盆子；看到刺状突起→金樱子。
- **桃仁 vs 苦杏仁**：桃仁扁长卵形、较大（长1.2~1.8cm）、表面密被纵皱纹；苦杏仁扁心形、较小（长1~1.9cm）、一端尖一端钝圆、左右不对称。心形不对称→苦杏仁；长卵形→桃仁。
- **地肤子 vs 小茴香**：地肤子扁球状五角星形、周围有5枚膜质小翅；小茴香是圆柱形双悬果、背面有5条纵棱线。五角星形带翅→地肤子；细长圆柱形→小茴香。
- **砂仁 / 豆蔻 vs 草豆蔻**：砂仁表面棕褐色、密生刺状突起；豆蔻表面黄白色至淡黄棕色、有3条纵向槽纹、果皮体轻质脆；草豆蔻是灰棕色种子团、有纵棱线及沟纹、分成3瓣。黄白色→豆蔻；刺状突起→砂仁；灰棕色种子团→草豆蔻。
- **乌梅 vs 山茱萸 / 木瓜 / 川楝子**：乌梅类球形、乌黑色/棕黑色、皱缩不平、核坚硬；山茱萸是不规则片状或囊状（无核）、紫红色至紫黑色、质柔润；木瓜多纵剖成两半、外皮紫红色、果肉红棕色细腻；川楝子类球形、金黄色至棕黄色、具深纵沟及小凹点。乌黑色皱缩球→乌梅；囊状无核→山茱萸；纵剖两半→木瓜；金黄色带纵沟→川楝子。
- **五味子 vs 补骨脂**：五味子红色/紫红色球形浆果、直径5~8mm、皱缩显油润；补骨脂肾形、黑色/黑褐色、极小（长仅3~5mm）、具网状皱纹。红色油润小球→五味子；黑色肾形小粒→补骨脂。"""

# 阶段一：品种识别（带药典形态特征表 + 混淆对区分要点）
HERB_IDENTIFY_PROMPT = """你是中药材鉴定专家，精通《中国药典》。

请观察图片，判断属于以下 20 种中药材中的哪一种。

## 候选品种及其药典形态特征
{herb_features}

## 易混淆品种对的区分要点（请重点对照）
{confusion_tips}

## 输出约束（严格 JSON）
```json
{{
  "herb_name": "从上述列表中选择的品种名",
  "confidence": 0.0-1.0,
  "reasoning": "简要说明识别依据，必须引用上面候选品种的具体形态特征（1-2句）"
}}
```

注意：
- herb_name 必须从上述列表中**精确选择**，不能写列表外的名字。
- 逐个核对候选品种的形态特征（形状/颜色/表面特征/大小），优先用最特异的特征区分。
- 如果看起来都不像，选最接近的那个，并把 confidence 降低。
- 严禁输出 JSON 以外的任何文本。"""

# 阶段二：定级推理（木桶效应 + 一票否决）
GRADING_SYSTEM_PROMPT = """你是国家认证的中药材商品规格等级鉴定专家，精通《中国药典》、SB/T 11173-2016《中药材商品规格等级通则》及 T/CACM 1021 系列团体标准。

你的任务：根据提供的【品种】、【定级规则】和【药材图片】，对样本进行质量等级判定。

## 定级原则（必须严格遵守）
1. **木桶效应**：样本等级由其最差维度对应的等级决定——任何一个维度不符合某等级标准，即不能评为该等级。
2. **一票否决**：若存在明显霉变、虫蛀、严重发黑/腐烂，直接定为等外品（D），无论其他维度多好。
3. **视觉优先**：以图片实际可见特征为准，不要凭空脑补。

## 工作流程
1. 观察图片，提取 4 个维度的观察值：color、size、plumpness、defect_rate。
2. 对照该品种提供的各档定级规则，逐项比对每个维度。
3. 按木桶效应取最低维度对应的等级。
4. 给出置信度（0-1），若低于阈值或品种识别存疑，将 review_flag 设为 true。

## 输出约束（严格 JSON）
```json
{{
  "grade": "{grade_tiers_str}",
  "grade_code": "{grade_code_str}",
  "confidence": 0.0-1.0,
  "evidence": {{
    "color": "颜色观察与匹配描述",
    "size": "尺寸/大小观察与匹配描述",
    "plumpness": "饱满度观察与匹配描述",
    "defect_rate": "缺陷/杂质/霉变/虫蛀观察与匹配描述"
  }},
  "lowest_dimension": "color|size|plumpness|defect_rate",
  "review_flag": true|false,
  "review_reason": "需要人工复核时填写，否则留空"
}}
```

注意：
- grade 和 grade_code 必须严格对应，等级映射如下：{grade_map_str}
- grade 只能从上述 {grade_tiers_str} 中选择，不能写其他等级名。
- lowest_dimension 必须填写导致降级的最弱维度。
- evidence 四个维度必须全部填写。
- 严禁输出 JSON 以外的任何文本。"""


def find_images(dataset_dir: Path) -> list[dict]:
    """
    遍历数据集目录。返回 [{true_herb, image_path}, ...]。
    文件夹名 = ground truth label（true_herb），不再喂给模型。
    """
    results = []
    if not dataset_dir.exists():
        print(f"[ERROR] 数据集目录不存在：{dataset_dir}")
        return results

    for herb_dir in sorted(dataset_dir.iterdir()):
        if not herb_dir.is_dir() or herb_dir.name.startswith("."):
            continue
        true_herb = herb_dir.name
        for img_path in sorted(herb_dir.iterdir()):
            if img_path.suffix.lower() in config.IMAGE_EXTENSIONS:
                results.append({"true_herb": true_herb, "image_path": str(img_path)})
    return results


def load_and_prepare_image(image_path: str, max_side: int = None) -> str:
    max_side = max_side or config.MAX_IMAGE_SIDE
    p = Path(image_path)
    if not p.exists():
        raise FileNotFoundError(image_path)

    if HAS_PIL:
        img = Image.open(p).convert("RGB")
        w, h = img.size
        if max(w, h) > max_side:
            ratio = max_side / max(w, h)
            img = img.resize((int(w * ratio), int(h * ratio)), Image.LANCZOS)
        buf = BytesIO()
        img.save(buf, format="JPEG", quality=90)
        data = buf.getvalue()
        mime = "image/jpeg"
    else:
        data = p.read_bytes()
        ext = p.suffix.lower()
        mime_map = {".jpg": "image/jpeg", ".jpeg": "image/jpeg",
                    ".png": "image/png", ".bmp": "image/bmp", ".webp": "image/webp"}
        mime = mime_map.get(ext, "image/jpeg")

    b64 = base64.b64encode(data).decode("ascii")
    return f"data:{mime};base64,{b64}"


def call_vision_api(system_prompt: str, user_prompt: str, image_data_url: str) -> str:
    if not config.OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY 未设置")

    try:
        from openai import OpenAI
        client = OpenAI(
            api_key=config.OPENAI_API_KEY,
            base_url=config.OPENAI_BASE_URL,
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt},
                    {"type": "image_url", "image_url": {"url": image_data_url}},
                ],
            },
        ]
        resp = client.chat.completions.create(
            model=config.VISION_MODEL,
            messages=messages,
            temperature=0.1,
            max_tokens=1024,
            response_format={"type": "json_object"},
        )
        return resp.choices[0].message.content

    except ImportError:
        import requests
        headers = {
            "Authorization": f"Bearer {config.OPENAI_API_KEY}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": config.VISION_MODEL,
            "temperature": 0.1,
            "max_tokens": 1024,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_prompt},
                        {"type": "image_url", "image_url": {"url": image_data_url}},
                    ],
                },
            ],
        }
        for attempt in range(config.MAX_RETRIES):
            resp = requests.post(
                f"{config.OPENAI_BASE_URL}/chat/completions",
                headers=headers, json=payload, timeout=120,
            )
            if resp.status_code == 200:
                return resp.json()["choices"][0]["message"]["content"]
            if resp.status_code >= 500:
                time.sleep(config.RETRY_DELAY * (attempt + 1))
                continue
            resp.raise_for_status()
        raise RuntimeError(f"API 调用失败，已重试 {config.MAX_RETRIES} 次")


def parse_llm_json(text: str) -> dict:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            pass
    return {}


def build_herb_features() -> dict:
    """从 rules.json 读取每个品种的药典形态描述，构建 {品种: 描述} 映射。"""
    rules_file = Path(__file__).parent / "data" / "rules.json"
    with open(rules_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {h["name_cn"]: h.get("pharmacopoeia_description", "") for h in data["herbs"]}


def identify_herb(image_data_url: str) -> dict:
    """阶段一：品种识别。带药典形态特征表 + 混淆对区分要点，纯看图选 20 种之一。"""
    herb_features = build_herb_features()
    herb_features_str = "\n".join(
        f"- {name}：{desc}" for name, desc in herb_features.items() if desc
    )
    sys_prompt = HERB_IDENTIFY_PROMPT.format(
        herb_features=herb_features_str,
        confusion_tips=CONFUSION_TIPS,
    )
    user_prompt = "请从上述 20 种中药材中选择最匹配的品种，输出 JSON。"

    for attempt in range(config.MAX_RETRIES):
        try:
            raw = call_vision_api(sys_prompt, user_prompt, image_data_url)
            result = parse_llm_json(raw)
            # 校验 herb_name 是否在列表里
            identified = result.get("herb_name", "")
            if identified and identified not in config.HERB_LIST:
                # 模糊匹配：取最接近的
                import difflib
                matches = difflib.get_close_matches(identified, config.HERB_LIST, n=1, cutoff=0.4)
                identified = matches[0] if matches else identified
            return {
                "identified_herb": identified,
                "herb_confidence": float(result.get("confidence", 0)),
                "herb_reasoning": result.get("reasoning", ""),
            }
        except Exception as e:
            if attempt < config.MAX_RETRIES - 1:
                time.sleep(config.RETRY_DELAY * (attempt + 1))
                continue
            return {
                "identified_herb": identified if 'identified' in dir() else "",
                "herb_confidence": 0.0,
                "herb_reasoning": f"API错误: {e}",
            }


def grade_herb(retriever, identified_herb: str, image_data_url: str) -> dict:
    """阶段二：用识别出的品种检索规则，再定级。"""
    hits = retriever.retrieve(identified_herb, query_text=None, top_k=8)
    if not hits:
        return {"status": "error", "error": f"未检索到 {identified_herb} 的定级规则"}

    # 拼接规则文本
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

    # 从检索到的规则里动态提取该品种的实际等级列表（不同品种等级体系可能不同）
    grade_tiers = [h["grade_tier"] for h in hits]
    grade_codes = [h.get("grade_code", "") for h in hits]
    grade_tiers_str = "|".join(grade_tiers)
    grade_code_str = "|".join(grade_codes)
    # 等级→code 的映射说明（注入 Prompt，让模型严格对齐）
    grade_map_str = "、".join(f"{t}→{c}" for t, c in zip(grade_tiers, grade_codes))

    sys_prompt = GRADING_SYSTEM_PROMPT.format(
        grade_tiers_str=grade_tiers_str,
        grade_code_str=grade_code_str,
        grade_map_str=grade_map_str,
    )

    user_prompt = (
        f"【已识别品种】{identified_herb}\n\n"
        f"【该品种定级规则】\n{rules_text}\n\n"
        f"【药材图片】见附件。请按木桶效应逐维度比对，输出 JSON。"
    )

    for attempt in range(config.MAX_RETRIES):
        try:
            raw = call_vision_api(sys_prompt, user_prompt, image_data_url)
            result = parse_llm_json(raw)
            if not result:
                raise ValueError("JSON 解析失败")
            # 置信度低于阈值 → review_flag
            conf = float(result.get("confidence", 0))
            if conf < config.REVIEW_CONFIDENCE_THRESHOLD:
                result["review_flag"] = True
                result["review_reason"] = (
                    result.get("review_reason", "")
                    + f" 置信度 {conf:.2f} < 阈值 {config.REVIEW_CONFIDENCE_THRESHOLD}"
                ).strip()
            return {"status": "ok", "result": result,
                    "retrieved_rules": [{"grade": h["grade_tier"], "grade_code": h.get("grade_code", ""), "score": h.get("score", 1.0)} for h in hits]}
        except Exception as e:
            if attempt < config.MAX_RETRIES - 1:
                time.sleep(config.RETRY_DELAY * (attempt + 1))
                continue
            return {"status": "error", "error": str(e)}


def process_single(retriever, true_herb: str, image_path: str) -> dict:
    """完整流程：图片 → 品种识别 → 定级 → 输出。"""
    image_data_url = load_and_prepare_image(image_path)

    # 阶段一：品种识别
    ident = identify_herb(image_data_url)
    identified_herb = ident["identified_herb"]
    herb_ok = identified_herb == true_herb

    # 阶段二：定级（无论识别对不对都跑，但识别错了定级结果不可靠）
    grade_result = grade_herb(retriever, identified_herb, image_data_url)

    # 合并输出
    if grade_result["status"] == "ok":
        r = grade_result["result"]
        # 如果品种识别错了，自动加 review_flag
        if not herb_ok:
            existing_reason = r.get("review_reason", "")
            r["review_flag"] = True
            r["review_reason"] = (existing_reason + " 品种识别错误，定级结果仅供参考").strip()
        return {
            "true_herb": true_herb,
            "identified_herb": identified_herb,
            "herb_confidence": ident["herb_confidence"],
            "herb_identification_correct": herb_ok,
            "image_path": image_path,
            "status": "ok",
            "result": r,
            "retrieved_rules": grade_result.get("retrieved_rules", []),
        }
    else:
        return {
            "true_herb": true_herb,
            "identified_herb": identified_herb,
            "herb_confidence": ident["herb_confidence"],
            "herb_identification_correct": herb_ok,
            "image_path": image_path,
            "status": "error",
            "error": grade_result.get("error", "定级失败"),
        }


def main():
    parser = argparse.ArgumentParser(description="Step3: 品种识别 + 定级推理（两阶段）")
    parser.add_argument("--mode", choices=["batch", "single"], default="batch")
    parser.add_argument("--image", type=str, help="single 模式图片路径")
    parser.add_argument("--herb", type=str, help="single 模式真实品种名（仅用于输出标签）")
    parser.add_argument("--dataset", type=Path, default=config.DATASET_DIR)
    parser.add_argument("--output", type=Path, default=config.RESULTS_DIR / "predictions.jsonl")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    from importlib.machinery import SourceFileLoader
    retriever_mod = SourceFileLoader("retriever", str(Path(__file__).parent / "02_retrieve_rules.py")).load_module()
    retriever = retriever_mod.RuleRetriever()

    args.output.parent.mkdir(parents=True, exist_ok=True)

    if args.mode == "single":
        if not args.image or not args.herb:
            parser.error("single 模式需要 --image 和 --herb")
        print(f"=== 单张模式：真实品种={args.herb} / {args.image}")
        result = process_single(retriever, args.herb, args.image)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        with open(args.output, "a", encoding="utf-8") as f:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
        return

    images = find_images(args.dataset)
    if not images:
        print(f"[WARN] 数据集目录 {args.dataset} 下未找到任何图片")
        return
    if args.limit:
        images = images[:args.limit]

    print(f"=== 批量模式：共 {len(images)} 张图片 ===")
    print(f"阶段一：品种识别（20种） → 阶段二：定级推理")
    t0 = time.time()
    ok_count = err_count = 0
    herb_correct = 0

    with open(args.output, "w", encoding="utf-8") as f:
        for i, item in enumerate(images, 1):
            true_herb = item["true_herb"]
            image_path = item["image_path"]
            print(f"[{i}/{len(images)}] {true_herb} | {Path(image_path).name} ... ", end="", flush=True)

            result = process_single(retriever, true_herb, image_path)
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
            f.flush()

            ident_ok = result.get("herb_identification_correct", False)
            ident_mark = "✅" if ident_ok else f"❌(真:{true_herb})"
            if ident_ok:
                herb_correct += 1

            if result["status"] == "ok":
                r = result["result"]
                print(f"识别→{ident_mark} → {r.get('grade', '?')} (conf={r.get('confidence', 0):.2f})"
                      f"{' ⚠需复核' if r.get('review_flag') else ''}")
                ok_count += 1
            else:
                print(f"识别→{ident_mark} → ERROR: {result.get('error', '?')}")
                err_count += 1

            if i % config.BATCH_SIZE == 0:
                time.sleep(0.5)

    elapsed = time.time() - t0
    herb_acc = herb_correct / len(images) * 100 if images else 0
    print(f"\n=== 完成：成功 {ok_count}，失败 {err_count}，耗时 {elapsed:.1f}s ===")
    print(f"品种识别准确率：{herb_acc:.1f}% ({herb_correct}/{len(images)})")
    print(f"结果已写入：{args.output}")


if __name__ == "__main__":
    main()
