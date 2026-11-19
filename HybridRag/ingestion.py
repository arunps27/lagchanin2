"""Ingestion: load docs -> chunk -> embed -> persist (Chroma + chunks.json for BM25)."""
import json
import shutil
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

from config import Settings, get_settings

_embedder_cache: dict[str, SentenceTransformer] = {}


def get_embedder(name: str) -> SentenceTransformer:
    if name not in _embedder_cache:
        _embedder_cache[name] = SentenceTransformer(name)
    return _embedder_cache[name]


def load_documents(data_dir: Path) -> list[dict]:
    docs = []
    for p in sorted(data_dir.glob("**/*")):
        ext = p.suffix.lower()
        if ext in {".txt", ".md"}:
            docs.append({"source": p.name, "text": p.read_text(encoding="utf-8")})
        elif ext == ".pdf":
            from pypdf import PdfReader
            try:
                text = "\n".join((pg.extract_text() or "") for pg in PdfReader(p).pages)
            except Exception as e:
                print(f"Skipping unreadable PDF {p.name}: {e}")
                continue
            if text.strip():
                docs.append({"source": p.name, "text": text})
            else:
                print(f"Skipping {p.name}: no extractable text (scanned PDF?)")
    return docs


def chunk_text(text: str, size: int, overlap: int) -> list[str]:
    words = text.split()
    step = max(1, size - overlap)
    chunks = []
    for start in range(0, len(words), step):
        piece = words[start:start + size]
        if piece:
            chunks.append(" ".join(piece))
        if start + size >= len(words):
            break
    return chunks


def ingest(settings: Settings | None = None) -> int:
    """Rebuild the index from scratch. Returns number of chunks indexed."""
    s = settings or get_settings()
    docs = load_documents(s.data_dir)
    if not docs:
        raise RuntimeError(f"No .txt/.md documents found in {s.data_dir}")

    chunks = []
    for d in docs:
        for i, c in enumerate(chunk_text(d["text"], s.chunk_words, s.chunk_overlap)):
            chunks.append({"id": f"{d['source']}::{i}", "source": d["source"], "text": c})

    embedder = get_embedder(s.embed_model)
    vectors = embedder.encode([c["text"] for c in chunks], normalize_embeddings=True,
                              batch_size=32, show_progress_bar=False)

    if s.index_dir.exists():
        shutil.rmtree(s.index_dir)
    s.index_dir.mkdir(parents=True)
    client = chromadb.PersistentClient(path=str(s.index_dir / "chroma"))
    col = client.create_collection(s.collection, metadata={"hnsw:space": "cosine"})
    col.add(ids=[c["id"] for c in chunks],
            embeddings=vectors.tolist(),
            documents=[c["text"] for c in chunks],
            metadatas=[{"source": c["source"]} for c in chunks])
    (s.index_dir / "chunks.json").write_text(json.dumps(chunks), encoding="utf-8")
    return len(chunks)


if __name__ == "__main__":
    print(f"Indexed {ingest()} chunks")
