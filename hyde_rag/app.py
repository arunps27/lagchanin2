"""Streamlit inspector. Run from the parent directory: streamlit run hyde_rag/app.py"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st  # noqa: E402

from hyde_rag.config import get_settings  # noqa: E402
from hyde_rag.evaluate import build_store  # noqa: E402
from hyde_rag.ingestion import VectorStore, load_directory, make_embedder  # noqa: E402
from hyde_rag.pipeline import RAGPipeline  # noqa: E402

st.set_page_config(page_title="Multi-Hypothesis HyDE RAG", layout="wide")
st.title("Multi-Hypothesis HyDE RAG")

with st.sidebar:
    st.header("Configuration")
    base = get_settings()
    api_key = st.text_input("LLM API key", base.llm_api_key, type="password")
    base_url = st.text_input("Base URL (Groq / OpenRouter / OpenAI)", base.llm_base_url)
    model = st.text_input("Model", base.llm_model)
    backend = st.selectbox("Embeddings", ["hash", "bge", "openai"], index=["hash", "bge", "openai"].index(base.embedding_backend))
    mode = st.radio("Mode", ["Multi-Hypothesis HyDE RAG", "Baseline RAG"])
    strategy = st.radio("Fusion strategy", ["rrf", "centroid"], format_func=lambda x: {"rrf": "A: Multi-vector + BM25 RRF", "centroid": "B: Centroid"}[x])
    top_k = st.slider("Top-K", 1, 10, base.top_k)
    rerank = st.checkbox("Cross-encoder rerank (bge-reranker-base)", False)
    docs_dir = st.text_input("Documents folder (blank = built-in synthetic corpus)", "")

settings = get_settings(llm_api_key=api_key, llm_base_url=base_url, llm_model=model, embedding_backend=backend, top_k=top_k, use_reranker=rerank, strategy=strategy)


@st.cache_resource(show_spinner="Indexing...")
def get_store(backend: str, docs_dir: str, key: str) -> VectorStore:
    s = get_settings(embedding_backend=backend, llm_api_key=key)
    if docs_dir.strip():
        return VectorStore(load_directory(Path(docs_dir), s), make_embedder(s))
    return build_store(s)


store = get_store(backend, docs_dir, api_key if backend == "openai" else "")
pipe = RAGPipeline(store, settings)
st.caption(f"{len(store.chunks)} chunks indexed" + ("" if docs_dir.strip() else " (built-in SYNTHETIC demo corpus)"))

query = st.text_input("Query", "Recent disclosure of EY for tax and compliance in India")
if st.button("Run", type="primary") and query:
    res = pipe.run(query, mode="hyde" if mode.startswith("Multi") else "baseline", strategy=strategy)
    st.subheader("Answer")
    st.markdown(res.answer)

    if res.hyde:
        with st.expander("Generated hypotheses", expanded=True):
            if res.hyde.fallback:
                st.warning(f"HyDE fell back to the raw query: {res.hyde.error}")
            for i, (h, a) in enumerate(zip(res.hyde.hypotheses, res.hyde.intent_angles), 1):
                st.markdown(f"**H{i} - {a}**")
                st.write(h)

    with st.expander("Retrieved chunks and scores", expanded=True):
        for h in res.hits:
            st.markdown(f"**{h.chunk.citation}** - score `{h.score:.4f}` - {h.detail}")
            st.write(h.chunk.text)

    with st.expander("Latency telemetry"):
        cols = st.columns(len(res.latency_ms))
        for c, (k, v) in zip(cols, res.latency_ms.items()):
            c.metric(k, f"{v:.0f} ms")
