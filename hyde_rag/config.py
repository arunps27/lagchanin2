"""Central settings. Reads env vars / .env via pydantic-settings if installed."""
from __future__ import annotations

from pathlib import Path

try:
    from pydantic_settings import BaseSettings, SettingsConfigDict

    _Base = BaseSettings
    _cfg = {"model_config": SettingsConfigDict(env_file=(Path(__file__).parent / ".env", ".env"), extra="ignore")}
except ImportError:  # minimal fallback so the package imports without pydantic-settings
    import os

    from pydantic import BaseModel

    class _Base(BaseModel):  # type: ignore[no-redef]
        def __init__(self, **kw):
            env = {k: v for k, v in os.environ.items()}
            for name in type(self).model_fields:
                if name not in kw and name.upper() in env:
                    kw[name] = env[name.upper()]
            super().__init__(**kw)

    _cfg = {}


class Settings(_Base):
    locals().update(_cfg)

    llm_api_key: str = ""
    llm_base_url: str = "https://api.groq.com/openai/v1"  # Groq / OpenRouter / OpenAI
    llm_model: str = "openai/gpt-oss-120b"
    llm_timeout: float = 30.0
    hyde_temperature: float = 0.3
    answer_temperature: float = 0.0
    n_hypotheses: int = 3

    # hash = offline deterministic (tests/eval); bge = sentence-transformers; openai = API
    embedding_backend: str = "hash"
    embedding_model: str = "BAAI/bge-base-en-v1.5"
    openai_embedding_model: str = "text-embedding-3-small"
    reranker_model: str = "BAAI/bge-reranker-base"
    use_reranker: bool = False

    chunk_size: int = 800
    chunk_overlap: int = 100
    top_k: int = 5
    candidate_k: int = 20
    rrf_k: int = 60
    strategy: str = "rrf"  # rrf (multi-vector) | centroid
    query_weight: float = 1.0  # centroid weight for the original query
    hypothesis_weight: float = 1.0  # centroid weight per hypothesis


def get_settings(**overrides) -> Settings:
    return Settings(**overrides)
