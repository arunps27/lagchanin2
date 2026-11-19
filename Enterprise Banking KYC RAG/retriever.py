"""Hybrid retrieval: BM25 + dense -> RRF -> cross-encoder rerank."""
import pickle
from typing import Dict, List

import chromadb
import numpy as np
from sentence_transformers import CrossEncoder, SentenceTransformer

from config import Settings, get_settings
from ingestion import tokenize


def rrf_fuse(rankings: List[List[str]], k: int = 60) -> List[tuple]:
    """Reciprocal Rank Fusion: score(d) = sum 1/(k + rank). Returns [(id, score)] desc."""
    scores: Dict[str, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


class HybridRetriever:
    def __init__(self, settings: Settings = None, embedder=None, reranker=None):
        self.s = settings or get_settings()
        with open(self.s.bm25_path, "rb") as fh:
            blob = pickle.load(fh)
        self.bm25 = blob["bm25"]
        self.chunks: Dict[str, Dict] = {c["id"]: c for c in blob["chunks"]}
        self.ids = [c["id"] for c in blob["chunks"]]
        self.embedder = embedder or SentenceTransformer(self.s.embed_model)
        self.reranker = reranker or CrossEncoder(self.s.rerank_model)
        self.col = chromadb.PersistentClient(path=self.s.chroma_dir).get_collection(self.s.collection)

    def bm25_search(self, query: str, k: int = None) -> List[str]:
        k = k or self.s.top_k_sparse
        scores = self.bm25.get_scores(tokenize(query))
        order = np.argsort(scores)[::-1][:k]
        return [self.ids[i] for i in order if scores[i] > 0]

    def dense_search(self, query: str, k: int = None) -> List[str]:
        k = k or self.s.top_k_dense
        qv = self.embedder.encode([self.s.query_instruction + query], normalize_embeddings=True)
        res = self.col.query(query_embeddings=qv.tolist(), n_results=min(k, self.col.count()))
        return res["ids"][0]

    def rerank(self, query: str, ids: List[str], top_k: int) -> List[tuple]:
        pairs = [(query, self.chunks[i]["text"]) for i in ids]
        scores = self.reranker.predict(pairs)
        ranked = sorted(zip(ids, scores), key=lambda x: x[1], reverse=True)[:top_k]
        return [(i, float(sc)) for i, sc in ranked]

    def retrieve(self, query: str) -> List[Dict]:
        sparse = self.bm25_search(query)
        dense = self.dense_search(query)
        fused = rrf_fuse([sparse, dense], self.s.rrf_k)[: self.s.rerank_candidates]
        fused_scores = dict(fused)
        reranked = self.rerank(query, [i for i, _ in fused], self.s.final_top_k)
        out = []
        for i, sc in reranked:
            c = self.chunks[i]
            out.append({
                "id": i, "text": c["text"], "source": c["source"], "section": c["section"],
                "rerank_score": sc, "rrf_score": fused_scores[i],
                "in_sparse": i in sparse, "in_dense": i in dense,
            })
        return out
