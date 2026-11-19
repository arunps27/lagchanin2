"""Vector-store ingestion pipeline:

Corpus --> Chunk --> Tokenization --> Embedding --> Indexing --> Vector DB Store

Run:  python ingest.py
"""
import statistics

import faiss  # noqa: F401  (ensures faiss is importable before langchain wraps it)
import numpy as np
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer

from config import EMBED_MAX_TOKENS, EMBED_MODEL, POLICY_DIR, VECTOR_STORE_DIR


def stage(n: int, title: str):
    print(f"\n[{n}/6] {title}")


def load_corpus() -> list[Document]:
    """Load .md / .txt / .pdf files from the policy folder (uploads land here too)."""
    docs = []
    for path in sorted(POLICY_DIR.iterdir()):
        ext = path.suffix.lower()
        if ext in (".md", ".txt"):
            text = path.read_text(encoding="utf-8", errors="replace")
        elif ext == ".pdf":
            from pypdf import PdfReader
            text = "\n".join((p.extract_text() or "") for p in PdfReader(str(path)).pages)
        else:
            continue
        if text.strip():
            docs.append(Document(page_content=text, metadata={"source": path.name}))
    return docs


def chunk(docs: list[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=80, separators=["\n## ", "\n\n", "\n", ". ", " "]
    )
    chunks = splitter.split_documents(docs)
    for i, c in enumerate(chunks):
        c.metadata["chunk_id"] = f"{c.metadata['source'].rsplit('.', 1)[0]}#{i:03d}"
    return chunks


def tokenize(chunks: list[Document], tokenizer) -> list[int]:
    """Count tokens with the embedding model's own tokenizer; re-split any over-long chunk."""
    counts = []
    for c in chunks:
        n = len(tokenizer.encode(c.page_content, add_special_tokens=True))
        if n > EMBED_MAX_TOKENS:
            print(f"   WARN {c.metadata['chunk_id']} has {n} tokens > {EMBED_MAX_TOKENS}; it will be truncated by the model")
        counts.append(n)
    return counts


def build_index(chunks: list[Document], embeddings: HuggingFaceEmbeddings) -> FAISS:
    texts = [c.page_content for c in chunks]
    vectors = np.array(embeddings.embed_documents(texts), dtype="float32")
    print(f"   embedded {vectors.shape[0]} chunks -> dim {vectors.shape[1]}")

    stage(5, "Indexing (FAISS IndexFlatIP over L2-normalized vectors = cosine)")
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    ids = [c.metadata["chunk_id"] for c in chunks]
    store = FAISS(
        embedding_function=embeddings,
        index=index,
        docstore=InMemoryDocstore({i: c for i, c in zip(ids, chunks)}),
        index_to_docstore_id={n: i for n, i in enumerate(ids)},
        distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT,
        normalize_L2=False,
    )
    print(f"   index.ntotal = {index.ntotal}")
    return store


def main():
    stage(1, "Corpus: load policy documents")
    docs = load_corpus()
    print(f"   {len(docs)} documents: {[d.metadata['source'] for d in docs]}")

    stage(2, "Chunk: recursive splitting (500 chars, 80 overlap)")
    chunks = chunk(docs)
    print(f"   {len(chunks)} chunks")

    stage(3, f"Tokenization: {EMBED_MODEL} tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL)
    counts = tokenize(chunks, tokenizer)
    print(f"   tokens/chunk: min={min(counts)} avg={statistics.mean(counts):.0f} max={max(counts)} (limit {EMBED_MAX_TOKENS})")

    stage(4, "Embedding: HuggingFace MiniLM (normalized)")
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL, encode_kwargs={"normalize_embeddings": True})
    store = build_index(chunks, embeddings)

    stage(6, f"Vector DB Store: persist to {VECTOR_STORE_DIR}")
    store.save_local(str(VECTOR_STORE_DIR))
    size = sum(p.stat().st_size for p in VECTOR_STORE_DIR.iterdir())
    print(f"   saved {len(list(VECTOR_STORE_DIR.iterdir()))} files, {size / 1024:.1f} KiB")
    print("\nIngestion complete.")


if __name__ == "__main__":
    main()
