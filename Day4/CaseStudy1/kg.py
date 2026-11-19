"""Knowledge graph over customers, identifiers and watchlists (NetworkX).

Graph side of Graph RAG: relationships (shared phone/address, fraud flags,
PEP family links, fuzzy sanctions matches) that a flat vector search misses.
"""
import json
import re
from difflib import SequenceMatcher

import networkx as nx

from config import DATA_DIR

IDENTIFIER_TYPES = {"Phone", "Address", "Account"}
SEVERITY_ORDER = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}


def _norm(name: str) -> str:
    tokens = re.sub(r"[^a-z\s]", " ", name.lower()).split()
    return " ".join(sorted(tokens))


def name_similarity(a: str, b: str) -> float:
    """Order-insensitive fuzzy name similarity in [0, 1]."""
    return SequenceMatcher(None, _norm(a), _norm(b)).ratio()


class KnowledgeGraph:
    def __init__(self, seed_path=None):
        seed = json.loads((seed_path or DATA_DIR / "graph_seed.json").read_text(encoding="utf-8"))
        self.g = nx.MultiDiGraph()
        for n in seed["nodes"]:
            attrs = {k: v for k, v in n.items() if k != "id"}
            self.g.add_node(n["id"], **attrs)
        for e in seed["edges"]:
            self.g.add_edge(e["src"], e["dst"], rel=e["rel"], detail=e.get("detail"))

    def ensure_person(self, person_id: str, name: str, dob: str | None = None):
        """Add an applicant not yet in the graph (e.g. uploaded applications)."""
        if person_id not in self.g:
            self.g.add_node(person_id, type="Person", name=name, dob=dob)

    # ---- helpers ----
    def label(self, nid: str) -> str:
        a = self.g.nodes[nid]
        return f"{a['type']}:{a['name']} ({nid})"

    def neighborhood_triples(self, entity_id: str, hops: int = 2) -> list[str]:
        """All edges within `hops` of the entity, as readable triples."""
        und = self.g.to_undirected(as_view=True)
        reach = nx.single_source_shortest_path_length(und, entity_id, cutoff=hops)
        triples = []
        for u, v, d in self.g.edges(data=True):
            if u in reach and v in reach:
                extra = f" [{d['detail']}]" if d.get("detail") else ""
                triples.append(f"{self.label(u)} -{d['rel']}{extra}-> {self.label(v)}")
        return sorted(triples)

    # ---- findings (each: type, severity, description, evidence, clause) ----
    def shared_identifier_findings(self, person_id: str) -> list[dict]:
        findings = []
        for _, ident, d in self.g.out_edges(person_id, data=True):
            if self.g.nodes[ident]["type"] not in IDENTIFIER_TYPES:
                continue
            others = [u for u, _, _ in self.g.in_edges(ident, data=True) if u != person_id]
            if not others:
                continue
            flagged = [o for o in others if self.g.nodes[o].get("fraud_flag")]
            flagged_acct = bool(self.g.nodes[ident].get("fraud_flag"))
            evidence = [f"{self.label(person_id)} -{d['rel']}-> {self.label(ident)}"]
            evidence += [f"{self.label(o)} -{d['rel']}-> {self.label(ident)}" for o in others]
            for o in flagged:
                evidence.append(f"FRAUD FLAG on {self.label(o)}: {self.g.nodes[o].get('fraud_note')}")
            sev = "HIGH" if (flagged or flagged_acct) else "MEDIUM"
            findings.append({
                "type": "SHARED_IDENTIFIER",
                "severity": sev,
                "description": (
                    f"{self.g.nodes[ident]['type']} '{self.g.nodes[ident]['name']}' is shared with "
                    f"{len(others)} other party(ies)"
                    + (", at least one with a confirmed fraud flag" if sev == "HIGH" else "")
                ),
                "evidence": evidence,
                "clause_hint": "KYC-3 shared identifier fraud ring rules",
            })
        return findings

    def watchlist_findings(self, name: str, dob: str | None, threshold: float = 0.85) -> list[dict]:
        findings = []
        for nid, a in self.g.nodes(data=True):
            if a["type"] != "WatchlistEntry":
                continue
            sim = name_similarity(name, a["name"])
            if sim < threshold:
                continue
            dob_match = bool(dob and a.get("dob") == dob)
            if a["category"] == "SANCTIONS":
                sev = "HIGH"
                hint = "AML-1 sanctions screening near match"
            else:
                sev = "MEDIUM"
                hint = "KYC-4 politically exposed persons"
            findings.append({
                "type": f"WATCHLIST_{a['category']}",
                "severity": sev,
                "description": (
                    f"Name similarity {sim:.0%} to {a['category']} entry '{a['name']}' "
                    f"({nid}); DOB {'MATCHES' if dob_match else 'does not match'}"
                ),
                "evidence": [f"{self.label(nid)} category={a['category']} dob={a.get('dob')}"],
                "clause_hint": hint,
            })
        return findings

    def pep_relationship_findings(self, person_id: str, hops: int = 2) -> list[dict]:
        """Persons within `hops` who are listed on a PEP/sanctions entry."""
        findings = []
        und = self.g.to_undirected(as_view=True)
        reach = nx.single_source_shortest_path_length(und, person_id, cutoff=hops)
        for nid in reach:
            if nid == person_id or self.g.nodes[nid]["type"] != "Person":
                continue
            for _, w, d in self.g.out_edges(nid, data=True):
                if d["rel"] != "LISTED_AS":
                    continue
                path = nx.shortest_path(und, person_id, nid)
                evidence = []
                for a, b in zip(path, path[1:]):
                    ed = self.g.get_edge_data(a, b) or self.g.get_edge_data(b, a)
                    r = next(iter(ed.values()))
                    evidence.append(f"{self.label(a)} -{r['rel']}[{r.get('detail')}]- {self.label(b)}")
                evidence.append(f"{self.label(nid)} -LISTED_AS-> {self.label(w)} ({self.g.nodes[w].get('note')})")
                cat = self.g.nodes[w]["category"]
                findings.append({
                    "type": f"RELATED_TO_{cat}",
                    "severity": "HIGH" if cat == "SANCTIONS" else "MEDIUM",
                    "description": f"Applicant is connected within {len(path) - 1} hop(s) to a {cat}-listed person",
                    "evidence": evidence,
                    "clause_hint": "KYC-4 PEP family or associate enhanced due diligence",
                })
        return findings

    def all_findings(self, person_id: str, name: str, dob: str | None) -> list[dict]:
        return (
            self.watchlist_findings(name, dob)
            + self.shared_identifier_findings(person_id)
            + self.pep_relationship_findings(person_id)
        )


def max_severity(findings: list[dict]) -> str:
    if not findings:
        return "LOW"
    return max((f["severity"] for f in findings), key=SEVERITY_ORDER.get)
