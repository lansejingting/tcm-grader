"""
RAG 向量库质量诊断脚本
======================
检查维度：
1. 向量分布检查（是否全挤在一坨 = 规则没区分度）
2. 品种内 vs 品种间相似度（同品种应该近，不同品种应该远）
3. 已知查询的检索效果（用规则里的文字做 query，看能否正确命中）
4. Embedding 模型合理性（重复 Chunks 向量是否一致）

用法: python 05_diagnose_rag.py
"""

import json
import sys
from pathlib import Path
from collections import defaultdict

import faiss
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.resolve()))
import config


def load_vectors_from_chunks(chunks):
    """从 chunks.json 的 text 字段重新 Embedding，获取向量矩阵。"""
    print("  正在重新 Embedding（从 chunks.json 的 text 字段）...")
    from openai import OpenAI
    client = OpenAI(
        api_key=config.EMBEDDING_API_KEY,
        base_url=config.EMBEDDING_BASE_URL,
    )

    texts = [c["text"] for c in chunks]
    all_vecs = []
    batch_size = 10
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        resp = client.embeddings.create(model=config.EMBEDDING_MODEL, input=batch)
        for d in resp.data:
            all_vecs.append(d.embedding)
        print(f"    [{min(i + batch_size, len(texts))}/{len(texts)}]")

    vecs = np.array(all_vecs, dtype=np.float32)
    # L2 归一化（FAISS IndexFlatIP 要求）
    faiss.normalize_L2(vecs)
    print(f"  ✅ 向量矩阵: {vecs.shape[0]} × {vecs.shape[1]}")
    return vecs


def check_vector_distribution(vecs, chunks):
    """检查向量分布：各等级是否真的拉开了距离。"""
    print("\n" + "=" * 60)
    print("【1】向量分布检查 —— 品种内不同等级的区分度")
    print("=" * 60)

    herb_indices = defaultdict(list)
    for i, c in enumerate(chunks):
        herb_indices[c["herb_name"]].append(i)

    print(f"\n  {'品种':<10s} {'向量数':>4s} {'等级间平均cosine距离':>22s} {'最小':>8s} {'最大':>8s}")
    print(f"  {'-'*56}")

    intra_dists = []
    for herb_name, indices in sorted(herb_indices.items()):
        v = vecs[indices]
        n = len(v)
        if n < 2:
            continue
        dists = []
        for i in range(n):
            for j in range(i + 1, n):
                sim = float(np.dot(v[i], v[j]))
                dists.append(1 - sim)
        avg_d = np.mean(dists)
        intra_dists.append(avg_d)
        print(f"  {herb_name:<10s} {n:>4d} {avg_d:>22.4f} {np.min(dists):>8.4f} {np.max(dists):>8.4f}")

    overall = np.mean(intra_dists)
    print(f"\n  整体品种内等级间平均距离: {overall:.4f}")
    if overall < 0.05:
        print("  ⚠ WARNING: 等级间距离极小，向量挤在一坨！规则内容太相似或 embedding 区分度不足。")
    elif overall < 0.15:
        print("  ⚠ WARNING: 等级间距离偏小，建议优化规则描述的区分度。")
    else:
        print("  ✅ 等级间距离合理，规则区分度良好。")


def check_cross_herb_similarity(vecs, chunks):
    """检查品种间相似度：不同品种应该差异大。"""
    print(f"\n{'='*60}")
    print("【2】品种内 vs 品种间相似度")
    print("=" * 60)

    herb_vecs = defaultdict(list)
    for i, c in enumerate(chunks):
        herb_vecs[c["herb_name"]].append(vecs[i])

    centroids = {}
    for herb, vs in herb_vecs.items():
        centroid = np.mean(vs, axis=0).copy()
        faiss.normalize_L2(centroid.reshape(1, -1))
        centroids[herb] = centroid

    herb_names = list(centroids.keys())

    # 品种内相似度
    intra_sims = []
    for herb_name, vs in herb_vecs.items():
        if len(vs) < 2:
            continue
        for i in range(len(vs)):
            for j in range(i + 1, len(vs)):
                intra_sims.append(float(np.dot(vs[i], vs[j])))

    # 品种间相似度
    inter_sims = []
    for i in range(len(herb_names)):
        for j in range(i + 1, len(herb_names)):
            inter_sims.append(float(np.dot(centroids[herb_names[i]], centroids[herb_names[j]])))

    intra_mean = np.mean(intra_sims) if intra_sims else 0
    inter_mean = np.mean(inter_sims) if inter_sims else 0

    print(f"\n  同品种内 cosine 相似度均值: {intra_mean:.4f}  (↑ 应该高)")
    print(f"  不同品种间 cosine 相似度均值: {inter_mean:.4f}  (↓ 应该低)")

    ratio = intra_mean / inter_mean if inter_mean > 0 else 0
    print(f"  内/外比值: {ratio:.2f}x  (> 1.5x 才合理)")

    if ratio < 1.2:
        print("  ⚠ WARNING: 内/外比值过低，品种间区分度差！")
    elif ratio < 1.5:
        print("  ⚠ WARNING: 品种区分度一般。")
    else:
        print("  ✅ 品种内相似度 > 品种间，区分度良好。")

    # 额外：列一个品种间相似度 Top 5（最容易混淆的品种对）
    print(f"\n  最容易混淆的品种对（相似度 Top 5）：")
    pairs = []
    for i in range(len(herb_names)):
        for j in range(i + 1, len(herb_names)):
            pairs.append((float(np.dot(centroids[herb_names[i]], centroids[herb_names[j]])),
                          herb_names[i], herb_names[j]))
    pairs.sort(reverse=True)
    for sim, h1, h2 in pairs[:5]:
        print(f"    {h1} ↔ {h2}: {sim:.4f}")


def check_retrieval_quality(vecs, chunks):
    """用规则里的真实文字做 query，看检索是否合理。"""
    print(f"\n{'='*60}")
    print("【3】已知查询检索测试")
    print("=" * 60)

    from openai import OpenAI
    client = OpenAI(
        api_key=config.EMBEDDING_API_KEY,
        base_url=config.EMBEDDING_BASE_URL,
    )

    # 建立 FAISS 索引
    index = faiss.IndexFlatIP(vecs.shape[1])
    index.add(vecs)

    herb_grades = defaultdict(list)
    for c in chunks:
        herb_grades[c["herb_name"]].append(c)

    test_count = 0
    hit_count = 0
    top3_hit = 0
    queries = []
    for herb_name in ["枸杞子", "乌梅", "五味子", "山楂", "连翘"]:
        if herb_name not in herb_grades:
            continue
        # 取一等品的 criteria 做 query（截前 150 字）
        grade_a = [c for c in herb_grades[herb_name]
                   if c.get("grade_code") in ("S", "A") or c.get("grade_tier") in ("特优级", "特级", "一等品")]
        if not grade_a:
            continue
        query_text = grade_a[0]["text"][:150]
        queries.append((herb_name, query_text))

    for herb_name, query_text in queries:
        resp = client.embeddings.create(model=config.EMBEDDING_MODEL, input=[query_text])
        q = np.array(resp.data[0].embedding, dtype=np.float32).reshape(1, -1)
        faiss.normalize_L2(q)

        scores, ids = index.search(q, min(5, index.ntotal))

        print(f"\n  Query: 【{herb_name}】{query_text[:50]}...")
        for rank, (score, idx) in enumerate(zip(scores[0], ids[0])):
            if idx < 0:
                continue
            c = chunks[int(idx)]
            match = "✅" if c["herb_name"] == herb_name else "❌"
            print(f"    {rank+1}. {match} {c['herb_name']}-{c['grade_tier']} score={float(score):.4f}")
            test_count += 1
            if c["herb_name"] == herb_name:
                hit_count += 1
                if rank < 3:
                    top3_hit += 1

    if test_count > 0:
        hit_rate = hit_count / test_count
        top3_rate = top3_hit / (len(queries) * 3)
        print(f"\n  品种命中率 (Top-5 全表): {hit_count}/{test_count} = {hit_rate:.2%}")
        print(f"  Top-3 品种命中率: {top3_rate:.2%}")
        if hit_rate < 0.5:
            print("  ⚠ WARNING: 检索经常跨品种！")
        elif hit_rate < 0.8:
            print("  ⚠ WARNING: 检索精度一般。")
        else:
            print("  ✅ 检索精度良好。")


def check_duplicate_vectors(vecs, chunks):
    """检查是否有重复 Chunk 或向量完全相同。"""
    print(f"\n{'='*60}")
    print("【4】重复向量检查")
    print("=" * 60)

    dup_count = 0
    near_dup = 0
    for i in range(len(vecs)):
        for j in range(i + 1, len(vecs)):
            sim = float(np.dot(vecs[i], vecs[j]))
            if sim > 0.9999:
                dup_count += 1
                if dup_count <= 3:
                    print(f"  重复: {chunks[i]['herb_name']}-{chunks[i]['grade_tier']} "
                          f"≈ {chunks[j]['herb_name']}-{chunks[j]['grade_tier']} sim={sim:.6f}")
            elif sim > 0.95:
                near_dup += 1

    if dup_count == 0 and near_dup == 0:
        print("  ✅ 无重复向量。")
    else:
        if dup_count > 0:
            print(f"  ⚠ 发现 {dup_count} 对几乎相同的向量 (sim>0.9999)")
        if near_dup > 0:
            print(f"  ⚠ 发现 {near_dup} 对高度相似向量 (0.95<sim<0.9999)")


def main():
    print("╔══════════════════════════════════════════════════════════╗")
    print("║          TCM-Grader RAG 向量库质量诊断                   ║")
    print("╚══════════════════════════════════════════════════════════╝")

    # 加载 chunks
    with open(config.CHUNKS_FILE, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    print(f"\nChunks 数量: {len(chunks)}")
    print(f"Chunks 文件: {config.CHUNKS_FILE}")

    # 重新 Embedding（避免 faiss 版本兼容问题）
    vecs = load_vectors_from_chunks(chunks)

    # 四项检查
    check_vector_distribution(vecs, chunks)
    check_cross_herb_similarity(vecs, chunks)
    check_retrieval_quality(vecs, chunks)
    check_duplicate_vectors(vecs, chunks)

    print(f"\n{'='*60}")
    print("诊断完成。✅=正常 ⚠=需关注")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
