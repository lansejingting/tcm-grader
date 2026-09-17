"""
Step 2: 规则检索模块
--------------------
给定药材名（herb_name），先在 FAISS 中做品种精确过滤，
再用语义检索取 Top-K 最相关的定级规则 Chunk。

可独立运行做调试：python 02_retrieve_rules.py --herb 枸杞子
也可被 Step3 作为模块导入：from 02_retrieve_rules import RuleRetriever
"""

import argparse
import json
import sys
from pathlib import Path

import faiss
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.resolve()))
import config


class RuleRetriever:
    """RAG 规则检索器：精确过滤 + 语义召回。"""

    def __init__(
        self,
        faiss_path: Path = None,
        chunks_path: Path = None,
    ):
        self.faiss_path = faiss_path or config.FAISS_INDEX_FILE
        self.chunks_path = chunks_path or config.CHUNKS_FILE
        self._load()

    def _load(self):
        if not self.faiss_path.exists():
            raise FileNotFoundError(
                f"FAISS 索引不存在：{self.faiss_path}\n"
                f"请先运行 Step1：python 01_build_knowledge_base.py"
            )
        if not self.chunks_path.exists():
            raise FileNotFoundError(f"Chunks 文件不存在：{self.chunks_path}")

        self.index = faiss.read_index(str(self.faiss_path))
        with open(self.chunks_path, "r", encoding="utf-8") as f:
            self.chunks = json.load(f)

        # 预建 herb_name → [chunk_indices] 映射，加速精确过滤
        self._herb_to_indices: dict[str, list[int]] = {}
        for idx, chunk in enumerate(self.chunks):
            # 中文名、英文名、拉丁名都作为可匹配 key
            for key_field in ("herb_name", "herb_en", "herb_latin"):
                key = chunk.get(key_field, "")
                if key:
                    self._herb_to_indices.setdefault(key, []).append(idx)

    def _embed_query(self, query_text: str) -> np.ndarray:
        """对查询文本做 embedding。"""
        try:
            from openai import OpenAI
            client = OpenAI(
                api_key=config.EMBEDDING_API_KEY,
                base_url=config.EMBEDDING_BASE_URL,
            )
            resp = client.embeddings.create(model=config.EMBEDDING_MODEL, input=[query_text])
            vec = np.array(resp.data[0].embedding, dtype=np.float32)
        except ImportError:
            import requests
            resp = requests.post(
                f"{config.EMBEDDING_BASE_URL}/embeddings",
                headers={
                    "Authorization": f"Bearer {config.EMBEDDING_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={"model": config.EMBEDDING_MODEL, "input": [query_text]},
                timeout=30,
            )
            resp.raise_for_status()
            vec = np.array(resp.json()["data"][0]["embedding"], dtype=np.float32)

        faiss.normalize_L2(vec.reshape(1, -1))
        return vec.reshape(1, -1)

    def retrieve(
        self,
        herb_name: str,
        query_text: str = None,
        top_k: int = None,
    ) -> list[dict]:
        """
        检索指定药材的定级规则。

        Args:
            herb_name: 药材中文名或别名
            query_text: 附加语义查询（可选）。若提供，先用精确过滤锁定品种，
                        再用语义从该品种内按 query_text 重排；若不提供，
                        直接返回该品种全部规则（按等级顺序）。
            top_k: 返回前 K 条，默认 config.RETRIEVAL_TOP_K

        Returns:
            list[dict] — 命中的规则 Chunk（含 score 字段表示相关度）
        """
        top_k = top_k or config.RETRIEVAL_TOP_K

        # Step A: 精确品种过滤
        candidate_indices = self._herb_to_indices.get(herb_name, [])
        if not candidate_indices:
            # 尝试模糊匹配（包含关系）
            for key in self._herb_to_indices:
                if herb_name in key or key in herb_name:
                    candidate_indices.extend(self._herb_to_indices[key])
            candidate_indices = list(set(candidate_indices))

        if not candidate_indices:
            print(f"[WARN] 未找到药材 '{herb_name}' 对应的规则，返回空列表")
            return []

        # Step B: 若无 query_text，直接按等级顺序返回
        if not query_text:
            hits = [self.chunks[i] for i in candidate_indices]
            # 优先用 grade_code 排序（S→A→B→C→D），回退 GRADE_TIERS 列表位置
            code_order = {c: i for i, c in enumerate(config.GRADE_CODE_ORDER)}
            hits.sort(key=lambda c: (
                code_order.get(c.get("grade_code", ""), 99),
                config.GRADE_TIERS.index(c["grade_tier"]) if c["grade_tier"] in config.GRADE_TIERS else 99,
            ))
            for h in hits:
                h["score"] = 1.0  # 精确匹配给满分
            return hits[:top_k]

        # Step C: 有 query_text → 语义重排
        query_vec = self._embed_query(query_text)
        # FAISS IndexFlatIP 暴露 xb；某些封装没有，做兜底
        try:
            all_vecs = faiss.vector_to_array(self.index.xb).reshape(self.index.ntotal, self.index.d)
        except Exception:
            # 无法直接取向量时，退化为全库 search 再按 candidate_indices 过滤
            scores, ids = self.index.search(query_vec, min(self.index.ntotal, top_k * 5))
            filtered = [(scores[0][j], ids[0][j]) for j in range(len(ids[0])) if ids[0][j] in candidate_indices]
            filtered.sort(key=lambda x: -x[0])
            hits = [self.chunks[int(idx)] for _, idx in filtered[:top_k]]
            for h, (s, _) in zip(hits, filtered[:top_k]):
                h["score"] = float(s)
            return hits

        # 有 xb → 只在候选集合内算点积
        cand_vecs = all_vecs[candidate_indices]
        faiss.normalize_L2(cand_vecs)
        scores = (cand_vecs @ query_vec.T).flatten()  # cosine similarity
        ranked = sorted(zip(scores, candidate_indices), key=lambda x: -x[0])
        hits = [self.chunks[idx] for _, idx in ranked[:top_k]]
        for h, (s, _) in zip(hits, ranked[:top_k]):
            h["score"] = float(s)
        return hits


def main():
    parser = argparse.ArgumentParser(description="Step2: 规则检索调试")
    parser.add_argument("--herb", required=True, help="药材名，如 '枸杞子'")
    parser.add_argument("--query", default=None, help="附加语义查询文本（可选）")
    parser.add_argument("--top-k", type=int, default=4, help="返回条数")
    args = parser.parse_args()

    retriever = RuleRetriever()
    hits = retriever.retrieve(args.herb, query_text=args.query, top_k=args.top_k)

    print(f"\n=== 检索结果：{args.herb}（query={args.query or '无'}）===")
    print(f"命中 {len(hits)} 条规则\n")
    for i, h in enumerate(hits, 1):
        code = h.get("grade_code", "")
        print(f"[{i}] (score={h.get('score', 0):.4f}) {h['herb_name']} - {h['grade_tier']}({code})")
        criteria = h.get("criteria", {})
        for dim, val in criteria.items():
            if val:
                print(f"    {dim}: {val}")
        print()


if __name__ == "__main__":
    main()
