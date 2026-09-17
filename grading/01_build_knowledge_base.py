"""
Step 1: 构建向量知识库
---------------------
将 rules.json 中的定级规则按"品种-等级-维度"拆分成 Chunk，
调用 Embedding 模型编码后存入 FAISS 索引，供 Step 2 检索使用。

输出：
    data/vector_store/faiss.index   — FAISS 向量索引
    data/vector_store/chunks.json   — 原始 Chunk 元数据（id → 规则文本）
"""

import argparse
import json
import sys
import time
from pathlib import Path

import faiss
import numpy as np

# 允许在脚本目录直接执行
sys.path.insert(0, str(Path(__file__).parent.resolve()))
import config


def load_rules(rules_path: Path) -> dict:
    """加载 rules.json，兼容新格式（herbs 为 list）。"""
    if not rules_path.exists():
        raise FileNotFoundError(f"规则文件不存在：{rules_path}")
    with open(rules_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if "herbs" not in data:
        raise ValueError("rules.json 缺少顶层 'herbs' 字段")

    # 新格式：herbs 是 list，每项含 name_cn / grades
    herbs_list = data["herbs"]
    if not isinstance(herbs_list, list):
        raise ValueError("rules.json 的 'herbs' 应为数组（list）格式")

    # 校验每个条目
    empty_count = 0
    for h in herbs_list:
        for g in h.get("grades", []):
            criteria = g.get("criteria", {})
            if all(not str(v).strip() for v in criteria.values()):
                empty_count += 1
    if empty_count > 0:
        print(f"[WARN] 检测到 {empty_count} 条等级规则的 criteria 为空，请检查填写完整性。")

    print(f"  加载 {len(herbs_list)} 个品种，标准来源：{data.get('metadata', {}).get('standard_sources', [])}")
    return data


def split_into_chunks(rules_data: dict) -> list[dict]:
    """
    将新格式 rules.json 拆分为 Chunk。
    Chunk 粒度：herb × grade → criteria + visual_conditions 完整描述
    """
    chunks = []
    herbs_list = rules_data["herbs"]

    for herb in herbs_list:
        herb_id = herb.get("id", "")
        herb_name = herb.get("name_cn", "")
        herb_en = herb.get("name_en", "")
        herb_latin = herb.get("name_latin", "")
        pharmacopoeia_desc = herb.get("pharmacopoeia_description", "")
        common_defects = herb.get("common_defects", [])
        visual_focus = herb.get("visual_focus", [])

        for grade_info in herb.get("grades", []):
            grade_tier = grade_info.get("grade", "")
            grade_code = grade_info.get("grade_code", "")
            criteria = grade_info.get("criteria", {})
            visual_conditions = grade_info.get("visual_conditions", {})

            # 组装 criteria 文本
            criteria_parts = []
            for dim_key, dim_val in criteria.items():
                if dim_val:
                    criteria_parts.append(f"【{dim_key}】{dim_val}")
            criteria_text = "；".join(criteria_parts) if criteria_parts else "（未填写）"

            # 组装 visual_conditions 文本（简化）
            vc_parts = []
            for vk, vv in visual_conditions.items():
                if isinstance(vv, list):
                    vc_parts.append(f"{vk}：{'/'.join(str(x) for x in vv)}")
                elif vv:
                    vc_parts.append(f"{vk}：{vv}")
            vc_text = "；".join(vc_parts)

            # 只放"等级差异"信息，去掉品种不变部分（药典描述等）
            # 品种信息已由文件名/ herb_name 精确过滤保证，不需要在向量里重复
            # 这样同一品种不同等级的向量才能拉开距离
            text = (
                f"药材：{herb_name}\n"
                f"等级：{grade_tier}（{grade_code}）\n"
                f"定级标准：{criteria_text}\n"
                f"视觉判定：{vc_text}"
            )

            chunks.append({
                "chunk_id": f"{herb_id}_{grade_code}",
                "herb_name": herb_name,
                "herb_id": herb_id,
                "herb_en": herb_en,
                "herb_latin": herb_latin,
                "grade_tier": grade_tier,
                "grade_code": grade_code,
                "criteria": criteria,
                "visual_conditions": visual_conditions,
                "text": text,
            })
    return chunks


def get_embeddings(texts: list[str], batch_size: int = 10) -> np.ndarray:
    """
    调用 Embedding API 批量编码文本。
    优先使用 openai SDK；若不可用则回退到 requests 直调。
    注意：DashScope Embedding API 单次最多 10 条，故默认 batch_size=10。
    """
    if not config.EMBEDDING_API_KEY:
        raise RuntimeError("EMBEDDING_API_KEY 未设置，请先配置环境变量")

    try:
        from openai import OpenAI
        client = OpenAI(
            api_key=config.EMBEDDING_API_KEY,
            base_url=config.EMBEDDING_BASE_URL,
        )
        embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            resp = client.embeddings.create(
                model=config.EMBEDDING_MODEL,
                input=batch,
            )
            embeddings.extend([d.embedding for d in resp.data])
            print(f"  Embedding 进度：{min(i + batch_size, len(texts))}/{len(texts)}")
        return np.array(embeddings, dtype=np.float32)

    except ImportError:
        # 回退：requests 直调
        import requests
        embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            resp = requests.post(
                f"{config.EMBEDDING_BASE_URL}/embeddings",
                headers={
                    "Authorization": f"Bearer {config.EMBEDDING_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={"model": config.EMBEDDING_MODEL, "input": batch},
                timeout=60,
            )
            resp.raise_for_status()
            data = resp.json()
            embeddings.extend([d["embedding"] for d in data["data"]])
            print(f"  Embedding 进度：{min(i + batch_size, len(texts))}/{len(texts)}")
        return np.array(embeddings, dtype=np.float32)


def build_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    """使用内积相似度（IP）构建 FAISS 索引，embedding 需先归一化。"""
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    # FAISS IndexFlatIP 假设输入已归一化；显式做一次防止遗漏
    faiss.normalize_L2(embeddings)
    index.add(embeddings)
    return index


def main():
    parser = argparse.ArgumentParser(description="Step1: 构建中药材定级规则向量知识库")
    parser.add_argument("--rules", type=Path, default=config.RULES_FILE, help="rules.json 路径")
    parser.add_argument("--output-dir", type=Path, default=config.VECTOR_STORE_DIR, help="向量库输出目录")
    parser.add_argument("--batch-size", type=int, default=10, help="Embedding 批大小（DashScope 最大 10）")
    args = parser.parse_args()

    # 1. 加载规则
    print(f"[1/4] 加载规则文件：{args.rules}")
    rules_data = load_rules(args.rules)

    # 2. 拆分 Chunk
    chunks = split_into_chunks(rules_data)
    n_herbs = len(rules_data["herbs"])
    n_grades_per_herb = len(rules_data["herbs"][0].get("grades", [])) if rules_data["herbs"] else 0
    print(f"[2/4] 拆分为 {len(chunks)} 个规则 Chunk（{n_herbs} 品种 × {n_grades_per_herb} 等级）")

    # 3. Embedding
    print(f"[3/4] 调用 Embedding 模型：{config.EMBEDDING_MODEL}")
    texts = [c["text"] for c in chunks]
    t0 = time.time()
    embeddings = get_embeddings(texts, batch_size=args.batch_size)
    print(f"  完成，耗时 {time.time() - t0:.1f}s，向量形状 {embeddings.shape}")

    # 4. 构建 FAISS 并持久化
    print(f"[4/4] 构建 FAISS 索引 → {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    index = build_faiss_index(embeddings)
    faiss.write_index(index, str(config.FAISS_INDEX_FILE))

    with open(config.CHUNKS_FILE, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

    print(f"  ✓ FAISS 索引：{config.FAISS_INDEX_FILE}")
    print(f"  ✓ Chunks 元数据：{config.CHUNKS_FILE}")
    print(f"  完成！共 {index.ntotal} 条规则向量化。")


if __name__ == "__main__":
    main()
