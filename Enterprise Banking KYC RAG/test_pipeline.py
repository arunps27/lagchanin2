"""End-to-end verification: Ingestion -> BM25 + Dense -> RRF -> Rerank -> Generation."""
import os
import shutil
import tempfile
from pathlib import Path
from types import SimpleNamespace

from config import get_settings
from generator import GroundedGenerator, build_prompt, extract_citations
from ingestion import ingest, tokenize
from retriever import HybridRetriever, rrf_fuse

SAMPLES = Path(__file__).parent / "tests" / "sample_docs"
CASES = [
    ("What ownership percentage makes someone a UBO?", "ubo_beneficial_ownership.md"),
    ("When is enhanced due diligence required for PEPs?", "edd_pep_monitoring.md"),
    ("How long must CDD records be retained?", "cdd_onboarding.md"),
]


def test_units():
    assert tokenize("The UBO owns 25%") == ["ubo", "owns", "25"]
    fused = rrf_fuse([["a", "b", "c"], ["c", "a", "d"]], k=60)
    ids = [i for i, _ in fused]
    assert len(ids) == len(set(ids)) == 4 and ids[0] in ("a", "c")
    assert extract_citations("Yes [1][3]. Also [1].") == [1, 3]
    chunks = [{"source": "x.md", "section": "S", "text": "hello"}]
    assert "[1] (source: x.md" in build_prompt("q?", chunks)


def test_end_to_end():
    tmp = Path(tempfile.mkdtemp())
    try:
        s = get_settings(data_dir=tmp)
        n = ingest(SAMPLES, s)
        assert n >= 3
        r = HybridRetriever(s)
        hits = 0
        for q, expected in CASES:
            sparse, dense = r.bm25_search(q), r.dense_search(q)
            assert sparse and dense, "both retrieval paths must return candidates"
            res = r.retrieve(q)
            assert 0 < len(res) <= s.final_top_k
            scores = [x["rerank_score"] for x in res]
            assert scores == sorted(scores, reverse=True)
            hits += any(x["source"] == expected for x in res)
            print(f"\nQ: {q}\n  top: {res[0]['source']} | {res[0]['section']} "
                  f"(rerank={scores[0]:.3f}, rrf={res[0]['rrf_score']:.4f})")
        assert hits == len(CASES), f"retrieval hit-rate {hits}/{len(CASES)}"

        # Offline generation check with a stub client (no network)
        fake = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
            create=lambda **kw: SimpleNamespace(choices=[SimpleNamespace(
                message=SimpleNamespace(content="A UBO owns 25% or more [1]."))]))))
        out = GroundedGenerator(s, client=fake).generate(CASES[0][0], r.retrieve(CASES[0][0]))
        assert out["grounded"] and out["citations"]

        if os.getenv("GROQ_API_KEY"):
            live = GroundedGenerator(s).generate(CASES[1][0], r.retrieve(CASES[1][0]))
            print("\nLIVE ANSWER:", live["answer"])
            assert live["mode"] == "documents" and not live["invalid_citations"]
            gen = "What is the difference between KYC and AML?"
            g = GroundedGenerator(s).generate(gen, r.retrieve(gen))
            print("\nGENERAL ANSWER:", g["mode"], "|", g["answer"][:200])
            assert g["mode"] in ("general", "mixed", "documents") and not g["invalid_citations"]
            off = "What is the capital of France?"
            o = GroundedGenerator(s).generate(off, r.retrieve(off))
            print("\nOFF-TOPIC:", o["mode"], "|", o["answer"][:120])
            assert o["mode"] == "out_of_scope"
        else:
            print("\n[skip] GROQ_API_KEY not set: live generation not exercised.")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    test_units()
    test_end_to_end()
    print("\nALL CHECKS PASSED")
