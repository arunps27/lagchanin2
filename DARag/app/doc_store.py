import re
from pathlib import Path

import chromadb
from chromadb.config import Settings as ChromaSettings


def chunk_markdown(text: str, source: str, max_chars: int = 900) -> list[dict]:
    """Heading-aware chunking: one chunk per section, long sections split on paragraphs."""
    sections: list[tuple[str, list[str]]] = []
    heading, buf = "Introduction", []
    for line in text.splitlines():
        m = re.match(r"^#{1,4}\s+(.*)", line)
        if m:
            if any(b.strip() for b in buf):
                sections.append((heading, buf))
            heading, buf = m.group(1).strip(), []
        else:
            buf.append(line)
    if any(b.strip() for b in buf):
        sections.append((heading, buf))

    chunks = []
    for heading, lines in sections:
        paras = [p.strip() for p in "\n".join(lines).split("\n\n") if p.strip()]
        cur = ""
        for p in paras:
            if cur and len(cur) + len(p) > max_chars:
                chunks.append((heading, cur))
                cur = p
            else:
                cur = f"{cur}\n\n{p}" if cur else p
        if cur:
            chunks.append((heading, cur))
    return [
        {"id": f"{source}::{i}", "text": f"{source} — {h}\n{t}", "source": source, "section": h}
        for i, (h, t) in enumerate(chunks)
    ]


class DocStore:
    """Policy document index backed by Chroma (default local ONNX MiniLM embeddings)."""

    def __init__(self, chroma_dir: Path, policies_dir: Path):
        self.policies_dir = Path(policies_dir)
        self.client = chromadb.PersistentClient(
            path=str(chroma_dir), settings=ChromaSettings(anonymized_telemetry=False)
        )
        self.col = self.client.get_or_create_collection("hr_policies", metadata={"hnsw:space": "cosine"})

    def count(self) -> int:
        return self.col.count()

    def _add_chunks(self, chunks: list[dict]):
        if not chunks:
            return
        self.col.upsert(
            ids=[c["id"] for c in chunks],
            documents=[c["text"] for c in chunks],
            metadatas=[{"source": c["source"], "section": c["section"]} for c in chunks],
        )

    def add_document(self, filename: str, text: str) -> int:
        existing = self.col.get(where={"source": filename})
        if existing["ids"]:
            self.col.delete(ids=existing["ids"])
        chunks = chunk_markdown(text, filename)
        self._add_chunks(chunks)
        return len(chunks)

    def reindex(self) -> dict:
        existing = self.col.get()
        if existing["ids"]:
            self.col.delete(ids=existing["ids"])
        files = sorted([*self.policies_dir.glob("*.md"), *self.policies_dir.glob("*.txt")])
        total = 0
        for f in files:
            total += self.add_document(f.name, f.read_text(encoding="utf-8"))
        return {"documents": len(files), "chunks": total}

    def search(self, query: str, k: int = 4) -> list[dict]:
        n = self.count()
        if n == 0:
            return []
        res = self.col.query(query_texts=[query], n_results=min(k, n))
        out = []
        for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
            out.append({"text": doc, "source": meta["source"], "section": meta["section"], "score": round(1 - dist, 3)})
        return out
