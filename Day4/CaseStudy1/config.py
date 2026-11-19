"""Central configuration: paths, model names, business thresholds, LLM factory."""
import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
POLICY_DIR = DATA_DIR / "policies"
VECTOR_STORE_DIR = BASE_DIR / "vector_store"

load_dotenv(BASE_DIR / ".env")

GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBED_MAX_TOKENS = 256

# Business thresholds (mirrored in data/policies/lending_policy.md)
MAX_DTI = 0.43
MIN_CREDIT_SCORE = 580
REFER_SCORE_BELOW = 680
MIN_HISTORY_YEARS = 2
MAX_LTV = 0.90
ASSUMED_APR = 0.07


def get_llm(temperature: float = 0.0):
    """Return ChatGroq bound to gpt-oss-120b. Key comes from GROQ_API_KEY."""
    from langchain_groq import ChatGroq

    if not os.getenv("GROQ_API_KEY"):
        raise RuntimeError("GROQ_API_KEY is not set. Copy .env.example to .env and add your key.")
    return ChatGroq(model=GROQ_MODEL, temperature=temperature, max_retries=3)
