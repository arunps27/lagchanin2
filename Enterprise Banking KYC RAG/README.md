# Enterprise KYC Hybrid RAG

BM25 + dense (bge-base-en-v1.5, Chroma) -> RRF (k=60) -> bge-reranker-base -> Groq `openai/gpt-oss-120b` with inline `[n]` citations.

## Setup
    python -m venv .venv && .venv\Scripts\activate
    pip install -r requirements.txt
    set GROQ_API_KEY=your_key      # or put it in .env (see .env.example)

## Run
    python test_pipeline.py
    python app.py ingest tests/sample_docs
    python app.py ask "When is enhanced due diligence required for PEPs?"

Modules: config.py, ingestion.py, retriever.py, generator.py, app.py. Swap Chroma for FAISS behind retriever.py to scale.
