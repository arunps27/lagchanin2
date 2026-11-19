"""Multi-hypothesis HyDE generation: structured JSON output, tenacity retries, fallback to the raw query."""
from __future__ import annotations

import json
import logging
import re

from pydantic import BaseModel, Field
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from .config import Settings

log = logging.getLogger(__name__)

SYSTEM_PROMPT = """You write hypothetical passages for retrieval (HyDE) in the domain of Indian tax and compliance.
Given a user query, write {n} DISTINCT hypothetical authoritative passages (80-140 words each), each answering from a
different interpretation (e.g. indirect tax/GST, direct tax/CBDT, corporate governance/regulatory filings). Mimic the
style of professional tax-advisory alerts. These passages are used only for similarity search and are never shown as facts.
Return ONLY JSON: {{"hypotheses": [{n} strings], "intent_angles": [{n} short labels]}}."""


class HydeOutput(BaseModel):
    hypotheses: list[str] = Field(min_length=1)
    intent_angles: list[str] = Field(default_factory=list)


class HydeResult(BaseModel):
    hypotheses: list[str]
    intent_angles: list[str]
    fallback: bool = False
    error: str | None = None


class TransientLLMError(Exception):
    """Timeouts / connection / rate-limit errors worth retrying."""


_TRANSIENT_NAMES = {"APITimeoutError", "APIConnectionError", "RateLimitError", "InternalServerError"}


def make_client(settings: Settings):
    from openai import OpenAI

    return OpenAI(api_key=settings.llm_api_key, base_url=settings.llm_base_url, timeout=settings.llm_timeout)


def parse_hyde_json(raw: str, n: int) -> HydeOutput:
    m = re.search(r"\{.*\}", raw, re.S)
    out = HydeOutput.model_validate(json.loads(m.group(0) if m else raw))
    out.hypotheses = [h.strip() for h in out.hypotheses if h.strip()][:n]
    if not out.hypotheses:
        raise ValueError("empty hypotheses")
    out.intent_angles = (out.intent_angles + [f"angle {i + 1}" for i in range(n)])[: len(out.hypotheses)]
    return out


class HydeGenerator:
    def __init__(self, settings: Settings, client=None):
        self.s = settings
        self.client = client  # injectable for tests; created lazily otherwise

    @retry(
        retry=retry_if_exception_type(TransientLLMError),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=0.5, max=4),
        reraise=True,
    )
    def _call(self, query: str) -> str:
        if self.client is None:
            self.client = make_client(self.s)
        try:
            r = self.client.chat.completions.create(
                model=self.s.llm_model,
                temperature=self.s.hyde_temperature,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT.format(n=self.s.n_hypotheses)},
                    {"role": "user", "content": query},
                ],
                response_format={"type": "json_object"},
            )
            return r.choices[0].message.content or ""
        except (TimeoutError, ConnectionError) as e:
            raise TransientLLMError(str(e)) from e
        except Exception as e:
            if type(e).__name__ in _TRANSIENT_NAMES:
                raise TransientLLMError(str(e)) from e
            raise

    def generate(self, query: str) -> HydeResult:
        try:
            out = parse_hyde_json(self._call(query), self.s.n_hypotheses)
            return HydeResult(hypotheses=out.hypotheses, intent_angles=out.intent_angles)
        except Exception as e:  # timeout, auth, parse/validation error -> degrade to raw query
            log.warning("HyDE fallback to raw query: %s", e)
            return HydeResult(hypotheses=[query], intent_angles=["raw query (fallback)"], fallback=True, error=f"{type(e).__name__}: {e}")
