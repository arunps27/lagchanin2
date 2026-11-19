"""CLI: `python app.py ingest <path>` | `python app.py ask "<question>"`."""
import argparse
import time

from config import get_settings


def main():
    p = argparse.ArgumentParser(description="Enterprise KYC Hybrid RAG")
    sub = p.add_subparsers(dest="cmd", required=True)
    pi = sub.add_parser("ingest")
    pi.add_argument("path")
    pa = sub.add_parser("ask")
    pa.add_argument("question")
    args = p.parse_args()
    s = get_settings()

    if args.cmd == "ingest":
        from ingestion import ingest
        print(f"Indexed {ingest(args.path, s)} chunks.")
        return

    from generator import GroundedGenerator
    from retriever import HybridRetriever
    t0 = time.perf_counter()
    chunks = HybridRetriever(s).retrieve(args.question)
    t1 = time.perf_counter()
    result = GroundedGenerator(s).generate(args.question, chunks)
    t2 = time.perf_counter()
    print(result["answer"], "\n\nSources:")
    for c in result["citations"]:
        print(f"  [{c['n']}] {c['source']} - {c['section']}")
    print(f"\n(retrieval {t1 - t0:.2f}s, generation {t2 - t1:.2f}s)")


if __name__ == "__main__":
    main()
