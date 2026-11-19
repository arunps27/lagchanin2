"""Central configuration. Secrets come from environment / .env only."""
import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).parent
load_dotenv(BASE_DIR / ".env")


@dataclass
class Settings:
    # --- paths ---
    data_dir: Path = BASE_DIR / "data"
    index_dir: Path = BASE_DIR / "index"
    collection: str = "hybrid_rag"

    # --- models ---
    embed_model: str = "BAAI/bge-base-en-v1.5"
    rerank_model: str = "BAAI/bge-reranker-base"
    query_prefix: str = "Represent this sentence for searching relevant passages: "

    # --- LLM (Groq, OpenAI-compatible) ---
    llm_base_url: str = "https://api.groq.com/openai/v1"
    llm_model: str = "openai/gpt-oss-120b"
    llm_api_key: str = os.getenv("GROQ_API_KEY", "")

    # --- chunking ---
    chunk_words: int = 120
    chunk_overlap: int = 30

    # --- the 4 tunable hyperparameters ---
    top_n: int = 20          # candidates per retriever (BM25 and dense)
    rrf_k: int = 60          # RRF damping constant
    top_k: int = 4           # final chunks after cross-encoder rerank
    temperature: float = 0.1  # LLM sampling temperature

    fused_top_n: int = 15    # fused candidates passed to the reranker


def get_settings() -> Settings:
    return Settings()
