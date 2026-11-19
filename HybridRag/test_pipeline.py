"""End-to-end smoke test: Ingestion -> BM25+Dense -> RRF -> Rerank -> Generation."""
import sys

from generator import generate
from ingestion import ingest
from retriever import HybridRetriever, rrf_fuse

# (question, expected source doc)
EVAL = [
    ("How many days of annual leave do employees get?", "hr_leave_policy.md"),
    ("What port does the WireGuard VPN use?", "vpn_remote_access.md"),
    ("Within how many hours must customers be notified of a data breach?", "security_incident_response.md"),
    ("What is the hotel cap per night for international travel?", "expense_reimbursement.md"),
    ("How long are application logs retained?", "data_retention_policy.md"),
]


def main() -> int:
    # RRF unit check
    f = rrf_fuse({"a": ["x", "y"], "b": ["y", "z"]}, 60)
    assert f["y"] > f["x"] and f["y"] > f["z"], "RRF should favour docs in both lists"
    print("[ok] RRF fusion")

    n = ingest()
    assert n > 0
    print(f"[ok] ingestion: {n} chunks")

    r = HybridRetriever()
    hits = 0
    for q, expected in EVAL:
        out = r.retrieve(q)
        res = out["results"]
        assert out["sparse_ids"] and out["dense_ids"], "both retrievers must return hits"
        scores = [x["rerank_score"] for x in res]
        assert scores == sorted(scores, reverse=True), "rerank order"
        hit = any(x["source"] == expected for x in res)
        hits += hit
        print(f"[{'ok' if hit else 'MISS'}] retrieval: {q} -> {res[0]['source']}")

    assert hits == len(EVAL), f"hit@k {hits}/{len(EVAL)}"

    q = EVAL[0][0]
    res = r.retrieve(q)["results"]
    gen = generate(q, res)
    print("\nAnswer:", gen["answer"])
    assert gen["cited"] and not gen["invalid"], f"bad citations: {gen}"
    print(f"[ok] generation with valid citations {gen['cited']}")
    print("\nALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
