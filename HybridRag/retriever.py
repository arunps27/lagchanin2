"""Hybrid retrieval: BM25 + dense -> RRF -> cross-encoder rerank."""
import json
import re
import time

import chromadb
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

from config import Settings, get_settings
from ingestion import get_embedder


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def rrf_fuse(rankings: dict[str, list[str]], k: int) -> dict[str, float]:
    """Reciprocal Rank Fusion: score(d) = sum_over_lists 1 / (k + rank)."""
    scores: dict[str, float] = {}
    for ids in rankings.values():
        for rank, cid in enumerate(ids, start=1):
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
    return scores


class HybridRetriever:
    def __init__(self, settings: Settings | None = None):
        self.s = settings or get_settings()
        chunks = json.loads((self.s.index_dir / "chunks.json").read_text(encoding="utf-8"))
        self.chunks = {c["id"]: c for c in chunks}
        self.ids = [c["id"] for c in chunks]
        self.bm25 = BM25Okapi([tokenize(c["text"]) for c in chunks])
        self.col = chromadb.PersistentClient(
            path=str(self.s.index_dir / "chroma")).get_collection(self.s.collection)
        self.embedder = get_embedder(self.s.embed_model)
        self.reranker = CrossEncoder(self.s.rerank_model)

    def sparse(self, query: str, n: int) -> list[str]:
        scores = self.bm25.get_scores(tokenize(query))
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:n]
        return [self.ids[i] for i in order if scores[i] > 0]

    def dense(self, query: str, n: int) -> list[str]:
        q = self.embedder.encode(self.s.query_prefix + query, normalize_embeddings=True)
        res = self.col.query(query_embeddings=[q.tolist()],
                             n_results=min(n, len(self.ids)))
        return res["ids"][0]

    def retrieve(self, query: str, top_n: int | None = None, rrf_k: int | None = None,
                 top_k: int | None = None) -> dict:
        top_n = top_n or self.s.top_n
        rrf_k = rrf_k or self.s.rrf_k
        top_k = top_k or self.s.top_k
        t = {}

        t0 = time.perf_counter()
        sparse_ids = self.sparse(query, top_n)
        t["bm25_ms"] = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        dense_ids = self.dense(query, top_n)
        t["dense_ms"] = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        fused = rrf_fuse({"bm25": sparse_ids, "dense": dense_ids}, rrf_k)
        fused_top = sorted(fused, key=fused.get, reverse=True)[: self.s.fused_top_n]
        t["rrf_ms"] = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        scores = self.reranker.predict([(query, self.chunks[c]["text"]) for c in fused_top])
        ranked = sorted(zip(fused_top, scores), key=lambda x: x[1], reverse=True)[:top_k]
        t["rerank_ms"] = (time.perf_counter() - t0) * 1000

        results = []
        for cid, sc in ranked:
            results.append({
                "id": cid,
                "source": self.chunks[cid]["source"],
                "text": self.chunks[cid]["text"],
                "rerank_score": float(sc),
                "rrf_score": fused[cid],
                "bm25_rank": sparse_ids.index(cid) + 1 if cid in sparse_ids else None,
                "dense_rank": dense_ids.index(cid) + 1 if cid in dense_ids else None,
            })
        return {"results": results, "timings": t,
                "sparse_ids": sparse_ids, "dense_ids": dense_ids, "fused_ids": fused_top}
