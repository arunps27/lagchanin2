from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    groq_api_key: str = ""
    llm_model: str = "openai/gpt-oss-120b"
    data_dir: Path = Path("data")
    max_upload_mb: int = 20
    max_result_rows: int = 5000
    rows_to_llm: int = 50
    allow_sensitive: bool = False
    retrieval_k: int = 4
    export_ttl_seconds: int = 900

    @property
    def db_path(self) -> Path:
        return self.data_dir / "hr.duckdb"

    @property
    def policies_dir(self) -> Path:
        return self.data_dir / "policies"

    @property
    def chroma_dir(self) -> Path:
        return self.data_dir / "chroma"

    @property
    def uploads_dir(self) -> Path:
        return self.data_dir / "uploads"


@lru_cache
def get_settings() -> Settings:
    return Settings()
