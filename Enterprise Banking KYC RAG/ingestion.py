"""Ingestion: load -> section-aware chunk -> tokenize (BM25) -> embed -> index."""
import pickle
import re
from pathlib import Path
from typing import Dict, List

import chromadb
from pypdf import PdfReader
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from config import Settings, get_settings

STOPWORDS = frozenset(
    "a an and are as at be but by for if in into is it no not of on or such that the "
    "their then there these they this to was will with".split()
)
_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[-'][a-z0-9]+)*")
_HEADING_RE = re.compile(r"^(#{1,6}\s+.+|\d+(\.\d+)*\.?\s+[A-Z].{0,80})$")


def tokenize(text: str) -> List[str]:
    """Lexical tokenizer for BM25: lowercase, alnum words, stopwords removed."""
    return [t for t in _TOKEN_RE.findall(text.lower()) if t not in STOPWORDS]


def load_documents(path: Path) -> List[Dict]:
    docs = []
    files = [path] if path.is_file() else sorted(p for p in path.rglob("*") if p.is_file())
    for f in files:
        ext = f.suffix.lower()
        if ext in (".txt", ".md"):
            text = f.read_text(encoding="utf-8", errors="ignore")
        elif ext == ".pdf":
            text = "\n".join((pg.extract_text() or "") for pg in PdfReader(str(f)).pages)
        else:
            continue
        if text.strip():
            docs.append({"source": f.name, "text": text})
    return docs


def chunk_document(doc: Dict, chunk_words: int, overlap: int) -> List[Dict]:
    """Split on headings first, then sliding word windows inside each section."""
    sections, heading, buf = [], "Introduction", []
    for line in doc["text"].splitlines():
        if _HEADING_RE.match(line.strip()):
            if buf:
                sections.append((heading, " ".join(buf)))
            heading, buf = line.strip().lstrip("# ").strip(), []
        elif line.strip():
            buf.append(line.strip())
    if buf:
        sections.append((heading, " ".join(buf)))

    chunks, step = [], max(1, chunk_words - overlap)
    for heading, body in sections:
        words = body.split()
        for i in range(0, max(1, len(words)), step):
            piece = " ".join(words[i:i + chunk_words])
            if piece:
                chunks.append({
                    "text": f"{heading}. {piece}",
                    "source": doc["source"],
                    "section": heading,
                })
            if i + chunk_words >= len(words):
                break
    return chunks


def ingest(path, settings: Settings = None, embedder: SentenceTransformer = None) -> int:
    s = settings or get_settings()
    docs = load_documents(Path(path))
    chunks: List[Dict] = []
    for d in docs:
        chunks.extend(chunk_document(d, s.chunk_words, s.chunk_overlap))
    if not chunks:
        raise ValueError(f"No ingestible documents found at {path}")
    for i, c in enumerate(chunks):
        c["id"] = f"chunk-{i}"

    embedder = embedder or SentenceTransformer(s.embed_model)
    vectors = embedder.encode([c["text"] for c in chunks], batch_size=32,
                              normalize_embeddings=True, show_progress_bar=False)

    s.data_dir.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=s.chroma_dir)
    try:
        client.delete_collection(s.collection)
    except Exception:
        pass
    col = client.create_collection(s.collection, metadata={"hnsw:space": "cosine"})
    col.add(
        ids=[c["id"] for c in chunks],
        embeddings=vectors.tolist(),
        documents=[c["text"] for c in chunks],
        metadatas=[{"source": c["source"], "section": c["section"]} for c in chunks],
    )

    bm25 = BM25Okapi([tokenize(c["text"]) for c in chunks])
    with open(s.bm25_path, "wb") as fh:
        pickle.dump({"bm25": bm25, "chunks": chunks}, fh)
    return len(chunks)
