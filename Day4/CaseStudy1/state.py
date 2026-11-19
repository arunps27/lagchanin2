from typing import Optional, TypedDict


class LoanState(TypedDict, total=False):
    application: dict
    intake: dict            # LLM-normalized summary + red flags from free-text notes
    graph_findings: list    # relationship findings from the knowledge graph
    graph_context: str      # 2-hop neighborhood triples (text)
    policy_chunks: list     # retrieved policy chunks [{chunk_id, text}]
    kyc: dict               # {risk, rule_floor, rationale, evidence, clauses}
    credit: dict            # deterministic metrics + rule outcome
    decision: str           # APPROVE | REFER | DECLINE | ESCALATE
    llm_decision: str       # what the LLM proposed before guardrails
    rationale: str
    conditions: list
    citations: dict
    needs_human: bool
    human_review: Optional[dict]
    audit_trail: list
