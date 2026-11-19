"""Retrieval benchmark: Baseline dense vs Single-HyDE vs Multi-HyDE + RRF.

All corpus text below is SYNTHETIC (written for testing; not real EY/CBIC/CBDT publications).
Without an LLM key, hypotheses come from canned per-query fixtures, so offline numbers
demonstrate the harness mechanics, not real-world HyDE quality. Set LLM_API_KEY to use live hypotheses.

Run: python -m hyde_rag.evaluate [--k 3] [--live]
"""
from __future__ import annotations

import argparse
import time

from .config import Settings, get_settings
from .hyde_generator import HydeGenerator
from .ingestion import Chunk, VectorStore, make_embedder
from .retriever import HydeRetriever

CORPUS: list[tuple[str, str, str]] = [
    ("gst_circular_einvoice.txt", "c1", "SYNTHETIC. GST circular on e-invoicing: mandatory e-invoice generation is extended to reverse charge supplies and import of services for registered taxpayers above the notified turnover threshold. Taxpayers must report invoices on the Invoice Registration Portal within the prescribed time limit."),
    ("gst_circular_refund.txt", "c2", "SYNTHETIC. CGST Rule 89 refund amendment: refund of unutilised input tax credit on account of inverted duty structure is clarified. The formula for maximum refund amount is revised and the refund application must be filed in Form GST RFD-01 with supporting statements."),
    ("gst_circular_isd.txt", "c3", "SYNTHETIC. Input Service Distributor (ISD) credit distribution norms: ISD must distribute input tax credit to recipient units in proportion to turnover, and must issue an ISD invoice. Credit on common services is distributed on a pro-rata basis within the month."),
    ("gst_registration.txt", "c4", "SYNTHETIC. GST registration and adjudication procedure: standardised timelines are prescribed for registration approval, show cause notices and adjudication orders. Taxpayers receive notices on the common portal and must respond within thirty days."),
    ("cbdt_foreign_assets.txt", "c5", "SYNTHETIC. Foreign assets disclosure scheme rules for small taxpayers: eligible resident taxpayers may declare undisclosed foreign assets or unreported overseas income by electronic filing of the prescribed forms before the scheme deadline. Fair market value of the asset is determined as per the prescribed valuation rules."),
    ("cbdt_foreign_assets_tax.txt", "c6", "SYNTHETIC. Disclosure scheme tax payment: tax and additional charge on declared foreign assets must be paid within the prescribed period after the declaration. A certificate of immunity is issued after payment and verification by the Income Tax Department."),
    ("cbdt_nri_reporting.txt", "c7", "SYNTHETIC. Cross-border income tax reporting for non-resident Indians: residential status tests, Schedule FA reporting of foreign assets in the income tax return, and treaty relief claimed through Form 67 for foreign tax credit."),
    ("transfer_pricing.txt", "c8", "SYNTHETIC. Transfer pricing documentation: entities with international transactions above the threshold must maintain contemporaneous documentation, file the accountant's report in Form 3CEB, and benchmark transactions at arm's length using the most appropriate method."),
    ("brsr_tax_transparency.txt", "c9", "SYNTHETIC. SEBI Business Responsibility and Sustainability Reporting (BRSR): listed companies disclose governance, tax transparency and sustainability metrics including the approach to tax and stakeholder engagement in the annual report."),
    ("fema_fcra.txt", "c10", "SYNTHETIC. FEMA and FCRA reporting amendments: Indian entities with foreign direct investment file the annual return on foreign liabilities and assets; FCRA registered entities must report foreign contributions quarterly and maintain a designated bank account."),
    ("internal_controls.txt", "c11", "SYNTHETIC. Internal financial controls and AI enabled compliance reporting: enterprises are expected to maintain a documented control framework over statutory reporting, with automated reconciliation and audit trails for compliance filings."),
    ("income_tax_slabs.txt", "c12", "SYNTHETIC. Individual income tax slabs: the new tax regime provides revised slab rates with a standard deduction for salaried individuals, and a rebate for resident individuals with income up to a specified limit."),
]

# (query, relevant chunk ids, canned HyDE hypotheses used offline)
GOLDEN: list[tuple[str, set[str], list[str]]] = [
    ("Recent disclosure of EY for tax and compliance in India", {"c1", "c5", "c9"},
     ["Tax alert on GST council circular: e-invoicing mandatory for reverse charge supplies, refund rule amendment and ISD credit distribution.",
      "Direct tax advisory: foreign assets disclosure scheme rules, electronic filing of forms, fair market value valuation and tax payment.",
      "Corporate compliance disclosures: transfer pricing documentation, BRSR tax transparency, FEMA and FCRA reporting, internal controls."]),
    ("How do small taxpayers declare undisclosed overseas holdings?", {"c5", "c6"},
     ["Scheme for small taxpayers to declare undisclosed foreign assets through electronic filing of prescribed forms and fair market value rules.",
      "Tax payment after declaration of foreign assets and immunity certificate under the disclosure scheme.",
      "Schedule FA reporting of foreign assets in the income tax return by resident taxpayers."]),
    ("Can refund of unused credit be claimed when output tax rate is lower?", {"c2"},
     ["Refund of unutilised input tax credit due to inverted duty structure under CGST Rule 89 with Form RFD-01 application.",
      "GST refund formula for inverted duty structure and supporting statements.",
      "Circular clarifying refund of accumulated input tax credit."]),
    ("What e-invoicing changes apply to import of services?", {"c1"},
     ["E-invoice generation mandatory for reverse charge supplies and import of services on the Invoice Registration Portal.",
      "GST circular extending e-invoicing to service imports above the turnover threshold.",
      "Invoice reporting timelines on the IRP for registered taxpayers."]),
    ("What documentation is needed for cross-border related party transactions?", {"c8"},
     ["Transfer pricing documentation and Form 3CEB accountant report for international transactions benchmarked at arm's length.",
      "Contemporaneous documentation requirement for associated enterprises above threshold.",
      "Most appropriate method selection and benchmarking study for related party transactions."]),
]


class CannedClient:
    """Fake LLM client returning fixed hypotheses (offline mode)."""

    def __init__(self, hyps: list[str]):
        import json

        self.payload = json.dumps({"hypotheses": hyps, "intent_angles": [f"angle {i + 1}" for i in range(len(hyps))]})
        self.chat = self
        self.completions = self

    def create(self, **_):
        from types import SimpleNamespace as NS

        return NS(choices=[NS(message=NS(content=self.payload))])


def metrics(ranked_ids: list[str], relevant: set[str], k: int) -> tuple[float, float]:
    top = ranked_ids[:k]
    recall = len(set(top) & relevant) / len(relevant)
    rr = next((1.0 / (i + 1) for i, c in enumerate(top) if c in relevant), 0.0)
    return recall, rr


def build_store(settings: Settings) -> VectorStore:
    return VectorStore([Chunk(cid, text, src) for src, cid, text in CORPUS], make_embedder(settings))


def run_benchmark(settings: Settings | None = None, k: int = 3, live: bool = False) -> dict[str, dict[str, float]]:
    s = settings or get_settings()
    store = build_store(s)
    retr = HydeRetriever(store, s)
    live = live and bool(s.llm_api_key)
    methods = {"Baseline Dense": [], "Single-HyDE": [], "Multi-HyDE + RRF": []}
    for q, rel, canned in GOLDEN:
        gen = HydeGenerator(s) if live else HydeGenerator(s, CannedClient(canned))
        t = time.perf_counter()
        base = [h.chunk.id for h in retr.baseline(q, k)]
        t_base = (time.perf_counter() - t) * 1000

        t = time.perf_counter()
        hyde = gen.generate(q)
        t_gen = (time.perf_counter() - t) * 1000

        t = time.perf_counter()
        single = [h.chunk.id for h in retr.retrieve(q, hyde.hypotheses[:1], "centroid", k)]
        t_single = (time.perf_counter() - t) * 1000 + t_gen

        t = time.perf_counter()
        multi = [h.chunk.id for h in retr.retrieve(q, hyde.hypotheses, "rrf", k)]
        t_multi = (time.perf_counter() - t) * 1000 + t_gen

        for name, ids, lat in (("Baseline Dense", base, t_base), ("Single-HyDE", single, t_single), ("Multi-HyDE + RRF", multi, t_multi)):
            r, rr = metrics(ids, rel, k)
            methods[name].append((r, rr, lat))
    n = len(GOLDEN)
    return {m: {"recall": sum(x[0] for x in v) / n, "mrr": sum(x[1] for x in v) / n, "latency_ms": sum(x[2] for x in v) / n} for m, v in methods.items()}


def print_table(res: dict[str, dict[str, float]], k: int) -> None:
    print(f"\n{'Method':<20}{f'Recall@{k}':>10}{'MRR':>8}{'Latency ms':>12}")
    print("-" * 50)
    for m, r in res.items():
        print(f"{m:<20}{r['recall']:>10.3f}{r['mrr']:>8.3f}{r['latency_ms']:>12.1f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--live", action="store_true", help="use real LLM for hypotheses (needs LLM_API_KEY)")
    a = ap.parse_args()
    s = get_settings()
    print(f"Embedder: {s.embedding_backend} | hypotheses: {'live LLM' if a.live and s.llm_api_key else 'canned fixtures (offline)'}")
    print_table(run_benchmark(s, a.k, a.live), a.k)
