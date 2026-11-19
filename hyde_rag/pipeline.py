"""End-to-end orchestrator (sync + async) with per-stage latency telemetry."""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field

from .config import Settings
from .generator import AnswerGenerator
from .hyde_generator import HydeGenerator, HydeResult
from .ingestion import VectorStore
from .retriever import Hit, HydeRetriever

log = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    query: str
    mode: str
    answer: str
    hits: list[Hit]
    hyde: HydeResult | None = None
    latency_ms: dict[str, float] = field(default_factory=dict)


class RAGPipeline:
    def __init__(self, store: VectorStore, settings: Settings, hyde: HydeGenerator | None = None, retriever: HydeRetriever | None = None, answerer: AnswerGenerator | None = None):
        self.s = settings
        self.hyde = hyde or HydeGenerator(settings)
        self.retriever = retriever or HydeRetriever(store, settings)
        self.answerer = answerer or AnswerGenerator(settings)

    def run(self, query: str, mode: str = "hyde", strategy: str | None = None, generate_answer: bool = True) -> PipelineResult:
        """mode: 'baseline' | 'hyde'."""
        lat: dict[str, float] = {}
        t0 = time.perf_counter()
        hyde = None
        if mode == "hyde":
            t = time.perf_counter()
            hyde = self.hyde.generate(query)
            lat["hyde_generation"] = (time.perf_counter() - t) * 1000
            t = time.perf_counter()
            hits = self.retriever.retrieve(query, hyde.hypotheses, strategy)
        else:
            t = time.perf_counter()
            hits = self.retriever.baseline(query)
        lat["retrieval"] = (time.perf_counter() - t) * 1000
        answer = ""
        if generate_answer:
            t = time.perf_counter()
            answer = self.answerer.answer(query, hits)
            lat["synthesis"] = (time.perf_counter() - t) * 1000
        lat["total"] = (time.perf_counter() - t0) * 1000
        log.info("query=%r mode=%s fallback=%s latency_ms=%s", query, mode, hyde.fallback if hyde else None, {k: round(v, 1) for k, v in lat.items()})
        return PipelineResult(query, mode, answer, hits, hyde, lat)

    async def arun(self, *args, **kwargs) -> PipelineResult:
        return await asyncio.to_thread(self.run, *args, **kwargs)
