"""Grounded answer synthesis with inline citations and refusal when context is insufficient."""
from __future__ import annotations

from .config import Settings
from .hyde_generator import make_client
from .retriever import Hit

SYSTEM = """You are a tax & compliance research assistant for India. Answer ONLY from the context chunks provided.
Rules:
1. Every factual sentence must end with an inline citation exactly like [Source: filename, Page/Chunk ID], copied from the chunk's label.
2. Never use outside knowledge. Do not invent circular numbers, dates, forms, thresholds or rates.
3. If the context does not contain the needed information, say so explicitly ("The retrieved documents do not contain ...") and list what is missing.
4. If only partly covered, answer the covered part and flag the gaps.
5. This is informational, not tax advice."""

REFUSAL = "The retrieved documents do not contain enough information to answer this question. No relevant regulatory context was found."


def format_context(hits: list[Hit]) -> str:
    return "\n\n".join(f"{h.chunk.citation}\n{h.chunk.text}" for h in hits)


class AnswerGenerator:
    def __init__(self, settings: Settings, client=None):
        self.s, self.client = settings, client

    def answer(self, query: str, hits: list[Hit]) -> str:
        if not hits:
            return REFUSAL
        if self.client is None:
            if not self.s.llm_api_key:
                return "(No LLM API key configured - showing retrieved context only.)\n\n" + format_context(hits)
            self.client = make_client(self.s)
        r = self.client.chat.completions.create(
            model=self.s.llm_model,
            temperature=self.s.answer_temperature,
            messages=[
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": f"Context:\n{format_context(hits)}\n\nQuestion: {query}"},
            ],
        )
        return r.choices[0].message.content or REFUSAL
