"""Streamlit developer UI for the Hybrid RAG pipeline."""
import time
from pathlib import Path

import streamlit as st

from config import get_settings
from generator import generate
from ingestion import ingest
from retriever import HybridRetriever

st.set_page_config(page_title="Hybrid RAG Console", layout="wide")
S = get_settings()


@st.cache_resource(show_spinner="Loading models...")
def load_retriever():
    return HybridRetriever(S)


st.title("Hybrid RAG Console")
st.caption("BM25 + Dense (bge-base) → RRF → Cross-Encoder (bge-reranker-base) → GPT-OSS-120B (Groq)")

with st.sidebar:
    st.header("Hyperparameters")
    top_n = st.slider("Candidates per retriever (top-N)", 5, 50, S.top_n)
    rrf_k = st.slider("RRF k constant", 1, 120, S.rrf_k)
    top_k = st.slider("Final chunks after rerank (top-K)", 1, 10, S.top_k)
    temperature = st.slider("LLM temperature", 0.0, 1.0, S.temperature, 0.05)
    st.divider()
    st.header("Documents")
    uploads = st.file_uploader("Upload PDF / TXT / MD", type=["pdf", "txt", "md"],
                               accept_multiple_files=True)
    if st.button("Upload & re-index", disabled=not uploads):
        S.data_dir.mkdir(exist_ok=True)
        for f in uploads:
            (S.data_dir / Path(f.name).name).write_bytes(f.getbuffer())
        with st.spinner("Indexing..."):
            n = ingest(S)
            st.cache_resource.clear()
        st.success(f"Saved {len(uploads)} file(s); indexed {n} chunks")
        st.rerun()
    st.caption(f"Indexed files: {', '.join(sorted(p.name for p in S.data_dir.glob('*') if p.is_file()))}")
    if st.button("Re-index documents"):
        with st.spinner("Indexing..."):
            n = ingest(S)
            st.cache_resource.clear()
        st.success(f"Indexed {n} chunks")

if not (S.index_dir / "chunks.json").exists():
    with st.spinner("First run: indexing sample documents..."):
        ingest(S)

retriever = load_retriever()
query = st.text_input("Ask a question about your documents",
                      "How many days of annual leave can be carried over?")

if st.button("Run", type="primary") and query.strip():
    t0 = time.perf_counter()
    ret = retriever.retrieve(query, top_n=top_n, rrf_k=rrf_k, top_k=top_k)
    chunks = ret["results"]
    try:
        gen = generate(query, chunks, temperature=temperature, settings=S)
    except Exception as e:  # surface API errors to the developer
        st.error(f"Generation failed: {e}")
        st.stop()
    total_ms = (time.perf_counter() - t0) * 1000
    tm = ret["timings"]

    both = sum(1 for c in chunks if c["bm25_rank"] and c["dense_rank"])
    overlap = both / len(chunks) if chunks else 0.0

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("End-to-end latency", f"{total_ms/1000:.2f} s")
    k2.metric("Top rerank score", f"{chunks[0]['rerank_score']:.2f}" if chunks else "–")
    k3.metric("Hybrid overlap", f"{overlap:.0%}")
    k4.metric("Citation coverage", f"{gen['coverage']:.0%}")
    st.caption(
        f"BM25 {tm['bm25_ms']:.0f} ms · Dense {tm['dense_ms']:.0f} ms · RRF {tm['rrf_ms']:.1f} ms · "
        f"Rerank {tm['rerank_ms']:.0f} ms · LLM {gen['llm_ms']:.0f} ms"
        + (f" · ⚠ invalid citations {gen['invalid']}" if gen["invalid"] else ""))

    st.subheader("Answer")
    st.write(gen["answer"])

    st.subheader("Sources")
    for i, c in enumerate(chunks, start=1):
        cited = "✅ cited" if i in gen["cited"] else "not cited"
        with st.expander(f"[{i}] {c['source']} — rerank {c['rerank_score']:.2f} ({cited})"):
            st.caption(f"BM25 rank: {c['bm25_rank'] or '–'} · Dense rank: {c['dense_rank'] or '–'} "
                       f"· RRF score: {c['rrf_score']:.4f}")
            st.write(c["text"])
