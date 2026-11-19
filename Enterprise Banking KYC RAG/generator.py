"""Grounded generation with strict inline citations via Groq (OpenAI-compatible)."""
import re
from typing import Dict, List

from openai import OpenAI

from config import Settings, get_settings

GENERAL_PREFIX = "General KYC knowledge (not from the indexed documents):"
OUT_OF_SCOPE = "This question is outside the scope of KYC/AML."

SYSTEM_PROMPT = (
    "You are a banking KYC/AML compliance assistant.\n"
    "1. If the numbered context sources answer the question, answer using them. Every "
    "sentence that relies on a source MUST end with inline citations like [1] or [1][3]. "
    "Never cite a number that was not provided.\n"
    "2. If the sources only partly answer it, use them for that part (with citations) and "
    "add the rest as general knowledge.\n"
    "3. If the sources do not answer a KYC/AML/CDD/EDD/sanctions/onboarding question, answer "
    f"from your general professional knowledge. Begin that answer with \"{GENERAL_PREFIX}\" "
    "and do NOT add [n] citations to general-knowledge statements. Be accurate, note that "
    "requirements vary by jurisdiction, and do not invent specific thresholds, article "
    "numbers or document names.\n"
    f"4. If the question is unrelated to KYC/AML/financial-crime compliance, reply exactly: "
    f"\"{OUT_OF_SCOPE}\""
)


def build_prompt(query: str, chunks: List[Dict]) -> str:
    ctx = "\n\n".join(
        f"[{i}] (source: {c['source']} | section: {c['section']})\n{c['text']}"
        for i, c in enumerate(chunks, start=1)
    )
    return f"Context sources:\n{ctx}\n\nQuestion: {query}\n\nAnswer with inline [n] citations:"


def normalize_answer(text: str) -> str:
    """Map full-width citation brackets to [n] and unusual spaces to plain spaces."""
    text = re.sub(r"[【\[［]\s*(\d+)\s*[】\]］]", r"[\1]", text)
    return re.sub(r"[      - ]", " ", text)


def extract_citations(answer: str) -> List[int]:
    return sorted({int(n) for n in re.findall(r"\[(\d+)\]", normalize_answer(answer))})


class GroundedGenerator:
    def __init__(self, settings: Settings = None, client: OpenAI = None):
        self.s = settings or get_settings()
        if client is None:
            if not self.s.groq_api_key:
                raise RuntimeError("GROQ_API_KEY is not set (env var or .env file).")
            client = OpenAI(api_key=self.s.groq_api_key, base_url=self.s.groq_base_url)
        self.client = client

    def generate(self, query: str, chunks: List[Dict]) -> Dict:
        resp = self.client.chat.completions.create(
            model=self.s.llm_model,
            temperature=self.s.llm_temperature,
            max_tokens=self.s.llm_max_tokens,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_prompt(query, chunks)},
            ],
        )
        answer = normalize_answer((resp.choices[0].message.content or "").strip())
        cited = extract_citations(answer)
        valid = [n for n in cited if 1 <= n <= len(chunks)]
        sources = [{"n": n, "source": chunks[n - 1]["source"], "section": chunks[n - 1]["section"]}
                   for n in valid]
        if answer.startswith(OUT_OF_SCOPE):
            mode = "out_of_scope"
        elif valid and GENERAL_PREFIX in answer:
            mode = "mixed"
        elif valid:
            mode = "documents"
        else:
            mode = "general"
        return {
            "answer": answer,
            "citations": sources,
            "invalid_citations": [n for n in cited if n not in valid],
            "mode": mode,
            "grounded": mode in ("documents", "mixed"),
        }
