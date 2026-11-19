"""Central configuration. Secrets come from the environment / .env only."""
import os
from dataclasses import dataclass, field
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:  # dotenv optional
    pass

BASE_DIR = Path(__file__).resolve().parent


@dataclass
class Settings:
    # LLM (Groq, OpenAI-compatible)
    groq_api_key: str = field(default_factory=lambda: os.getenv("GROQ_API_KEY", ""))
    groq_base_url: str = "https://api.groq.com/openai/v1"
    llm_model: str = "openai/gpt-oss-120b"
    llm_temperature: float = 0.1
    llm_max_tokens: int = 1024

    # Models
    embed_model: str = "BAAI/bge-base-en-v1.5"
    rerank_model: str = "BAAI/bge-reranker-base"
    query_instruction: str = "Represent this sentence for searching relevant passages: "

    # Chunking (words)
    chunk_words: int = 300
    chunk_overlap: int = 50

    # Retrieval
    top_k_sparse: int = 20
    top_k_dense: int = 20
    rrf_k: int = 60
    rerank_candidates: int = 20
    final_top_k: int = 5

    # Storage
    data_dir: Path = BASE_DIR / "data"
    collection: str = "kyc_chunks"

    @property
    def chroma_dir(self) -> str:
        return str(self.data_dir / "chroma")

    @property
    def bm25_path(self) -> Path:
        return self.data_dir / "bm25.pkl"


def get_settings(**overrides) -> Settings:
    s = Settings()
    for k, v in overrides.items():
        setattr(s, k, v)
    return s
