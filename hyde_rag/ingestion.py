"""Document loading, recursive chunking, embedding backends and an in-memory vector + BM25 index."""
from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .config import Settings

TOKEN_RE = re.compile(r"[a-z0-9]+")
STOP = {"the", "a", "an", "of", "and", "or", "to", "in", "for", "on", "by", "is", "are", "with", "as", "at", "from", "that", "this", "it", "be"}


def tokenize(text: str) -> list[str]:
    return [t for t in TOKEN_RE.findall(text.lower()) if t not in STOP]


@dataclass
class Chunk:
    id: str
    text: str
    source: str
    page: int | None = None
    meta: dict = field(default_factory=dict)

    @property
    def citation(self) -> str:
        loc = f"Page {self.page}" if self.page else f"Chunk {self.id}"
        return f"[Source: {self.source}, {loc}]"


# ---------------- loading & chunking ----------------
def load_file(path: Path) -> list[tuple[str, int | None]]:
    """Return [(text, page)] for a file. PDF/DOCX need optional deps."""
    ext = path.suffix.lower()
    if ext in {".txt", ".md"}:
        return [(path.read_text(encoding="utf-8", errors="ignore"), None)]
    if ext == ".pdf":
        from pypdf import PdfReader

        return [(p.extract_text() or "", i + 1) for i, p in enumerate(PdfReader(str(path)).pages)]
    if ext == ".docx":
        import docx

        return [("\n".join(p.text for p in docx.Document(str(path)).paragraphs), None)]
    raise ValueError(f"Unsupported file type: {path}")


_SEPS = ("\n\n", "\n", ". ", " ")


def recursive_split(text: str, size: int, overlap: int, seps: tuple[str, ...] = _SEPS) -> list[str]:
    """Split on the coarsest separator that fits, recursing into oversize pieces; adds char overlap."""
    text = text.strip()
    if len(text) <= size:
        return [text] if text else []
    if not seps:  # hard split
        step = max(size - overlap, 1)
        return [text[i : i + size] for i in range(0, len(text), step)]
    sep, finer = seps[0], seps[1:]
    if sep not in text:
        return recursive_split(text, size, overlap, finer)
    pieces: list[str] = []
    for p in text.split(sep):
        pieces.extend([p] if len(p) <= size else recursive_split(p, size, overlap, finer))
    out, cur = [], ""
    for p in pieces:
        if not cur:
            cur = p
        elif len(cur) + len(sep) + len(p) <= size:
            cur = cur + sep + p
        else:
            out.append(cur)
            tail = cur[-overlap:] if overlap else ""
            cur = (tail + sep + p) if tail and len(tail) + len(sep) + len(p) <= size else p
    if cur:
        out.append(cur)
    return [c.strip() for c in out if c.strip()]


def chunk_documents(paths: list[Path], settings: Settings) -> list[Chunk]:
    chunks: list[Chunk] = []
    for path in paths:
        for text, page in load_file(path):
            for j, piece in enumerate(recursive_split(text, settings.chunk_size, settings.chunk_overlap)):
                chunks.append(Chunk(f"{path.stem}-{page or 0}-{j}", piece, path.name, page))
    return chunks


def load_directory(directory: Path, settings: Settings) -> list[Chunk]:
    exts = {".txt", ".md", ".pdf", ".docx"}
    return chunk_documents(sorted(p for p in Path(directory).rglob("*") if p.suffix.lower() in exts), settings)


# ---------------- embeddings ----------------
def _l2(m: np.ndarray) -> np.ndarray:
    return m / np.maximum(np.linalg.norm(m, axis=-1, keepdims=True), 1e-12)


class Embedder:
    name = "base"

    def embed(self, texts: list[str], is_query: bool = False) -> np.ndarray:
        raise NotImplementedError


class HashEmbedder(Embedder):
    """Deterministic offline unigram+bigram hashing embedder (tests/eval without model downloads)."""

    name = "hash"

    def __init__(self, dim: int = 2048):
        self.dim = dim

    def embed(self, texts, is_query=False):
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            toks = tokenize(t)
            for f in toks + [f"{a}_{b}" for a, b in zip(toks, toks[1:])]:
                out[i, int(hashlib.md5(f.encode()).hexdigest(), 16) % self.dim] += 1.0
        return _l2(out)


class BGEEmbedder(Embedder):
    name = "bge"
    QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

    def __init__(self, model: str):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model)

    def embed(self, texts, is_query=False):
        texts = [self.QUERY_PREFIX + t for t in texts] if is_query else texts
        return np.asarray(self.model.encode(texts, normalize_embeddings=True), dtype=np.float32)


class OpenAIEmbedder(Embedder):
    name = "openai"

    def __init__(self, settings: Settings):
        from openai import OpenAI

        self.client = OpenAI(api_key=settings.llm_api_key, base_url=settings.llm_base_url, timeout=settings.llm_timeout)
        self.model = settings.openai_embedding_model

    def embed(self, texts, is_query=False):
        r = self.client.embeddings.create(model=self.model, input=texts)
        return _l2(np.array([d.embedding for d in r.data], dtype=np.float32))


def make_embedder(settings: Settings) -> Embedder:
    if settings.embedding_backend == "bge":
        return BGEEmbedder(settings.embedding_model)
    if settings.embedding_backend == "openai":
        return OpenAIEmbedder(settings)
    return HashEmbedder()


# ---------------- indexes ----------------
class BM25:
    def __init__(self, docs: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b, self.docs = k1, b, docs
        self.avgdl = sum(map(len, docs)) / max(len(docs), 1)
        df: dict[str, int] = {}
        for d in docs:
            for t in set(d):
                df[t] = df.get(t, 0) + 1
        n = len(docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}
        self.tf = [{t: d.count(t) for t in set(d)} for d in docs]

    def scores(self, query: list[str]) -> np.ndarray:
        s = np.zeros(len(self.docs))
        for i, (tf, d) in enumerate(zip(self.tf, self.docs)):
            norm = self.k1 * (1 - self.b + self.b * len(d) / (self.avgdl or 1))
            for t in query:
                f = tf.get(t)
                if f:
                    s[i] += self.idf[t] * f * (self.k1 + 1) / (f + norm)
        return s


class VectorStore:
    """In-memory dense + BM25 index over chunks."""

    def __init__(self, chunks: list[Chunk], embedder: Embedder):
        self.chunks, self.embedder = chunks, embedder
        self.matrix = embedder.embed([c.text for c in chunks]) if chunks else np.zeros((0, 1), np.float32)
        self.bm25 = BM25([tokenize(c.text) for c in chunks])

    def dense_search(self, vec: np.ndarray, k: int) -> list[tuple[int, float]]:
        sims = self.matrix @ vec
        return [(int(i), float(sims[i])) for i in np.argsort(-sims)[:k]]

    def bm25_search(self, query: str, k: int) -> list[tuple[int, float]]:
        s = self.bm25.scores(tokenize(query))
        return [(int(i), float(s[i])) for i in np.argsort(-s)[:k] if s[i] > 0]
