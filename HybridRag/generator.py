"""Grounded generation via Groq's OpenAI-compatible API with strict citations."""
import re
import time

from openai import OpenAI

from config import Settings, get_settings

SYSTEM_PROMPT = (
    "You are an enterprise knowledge assistant. Answer ONLY using the numbered "
    "context passages provided. Every sentence containing a fact must end with an "
    "inline citation like [1] or [1][3], using the passage numbers. Never cite a "
    "number that is not in the context. If the context does not contain the answer, "
    "reply exactly: \"I don't have enough information in the provided documents.\" "
    "Be concise."
)


def build_context(chunks: list[dict]) -> str:
    return "\n\n".join(f"[{i}] (source: {c['source']})\n{c['text']}"
                       for i, c in enumerate(chunks, start=1))


def citation_stats(answer: str, n_chunks: int) -> dict:
    cited = [int(x) for x in re.findall(r"\[(\d+)\]", answer)]
    valid = [c for c in cited if 1 <= c <= n_chunks]
    sentences = [x for x in re.split(r"(?<=[.!?])\s+", answer.strip()) if x]
    with_cite = [x for x in sentences if re.search(r"\[\d+\]", x)]
    return {
        "cited": sorted(set(valid)),
        "invalid": sorted(set(cited) - set(valid)),
        "coverage": (len(with_cite) / len(sentences)) if sentences else 0.0,
    }


def generate(query: str, chunks: list[dict], temperature: float | None = None,
             settings: Settings | None = None) -> dict:
    s = settings or get_settings()
    if not s.llm_api_key:
        raise RuntimeError("GROQ_API_KEY is not set (put it in .env)")
    client = OpenAI(api_key=s.llm_api_key, base_url=s.llm_base_url)
    t0 = time.perf_counter()
    resp = client.chat.completions.create(
        model=s.llm_model,
        temperature=s.temperature if temperature is None else temperature,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",
             "content": f"Context:\n{build_context(chunks)}\n\nQuestion: {query}"},
        ],
    )
    answer = resp.choices[0].message.content or ""
    return {"answer": answer, "llm_ms": (time.perf_counter() - t0) * 1000,
            **citation_stats(answer, len(chunks))}
