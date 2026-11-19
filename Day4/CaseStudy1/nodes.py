"""LangGraph nodes for the loan processing / KYC triage workflow."""
import json
from typing import Literal

from langchain_core.prompts import ChatPromptTemplate
from langgraph.types import interrupt
from pydantic import BaseModel, Field

import config
from kg import SEVERITY_ORDER, KnowledgeGraph, max_severity
from state import LoanState

Risk = Literal["LOW", "MEDIUM", "HIGH"]


# ---------- structured LLM outputs ----------
class IntakeResult(BaseModel):
    summary: str = Field(description="One-sentence neutral summary of the application")
    red_flags: list[str] = Field(default_factory=list, description="Risk indicators stated or implied in the notes")


class KycAssessment(BaseModel):
    risk: Risk
    rationale: str = Field(description="Concise reasoning grounded in the evidence")
    cited_graph_evidence: list[str] = Field(default_factory=list, description="Graph triples relied on, verbatim")
    cited_policy_clauses: list[str] = Field(default_factory=list, description="Clause IDs like KYC-3.2")


class DecisionResult(BaseModel):
    decision: Literal["APPROVE", "REFER", "DECLINE"]
    rationale: str
    conditions: list[str] = Field(default_factory=list)
    cited_policy_clauses: list[str] = Field(default_factory=list)


def _audit(state: LoanState, step: str, detail: str) -> list:
    return state.get("audit_trail", []) + [{"step": step, "detail": detail}]


def monthly_payment(principal: float, months: int, apr: float = config.ASSUMED_APR) -> float:
    r = apr / 12
    return principal / months if r == 0 else principal * r / (1 - (1 + r) ** -months)


def credit_metrics(app: dict) -> dict:
    """Deterministic credit rules (LEND policy). Returns metrics, flags and rule outcome."""
    pay = monthly_payment(app["loan_amount"], app["loan_term_months"])
    dti = (app["existing_monthly_debt"] + pay) / app["monthly_income"]
    ltv = app["loan_amount"] / app["property_value"] if app.get("property_value") else None
    score, hist = app["credit_score"], app["credit_history_years"]
    outcome, reasons = "APPROVE", []

    def worsen(new, why, clause):
        nonlocal outcome
        order = {"APPROVE": 0, "REFER": 1, "DECLINE": 2}
        if order[new] > order[outcome]:
            outcome = new
        reasons.append(f"{why} [{clause}]")

    if dti > config.MAX_DTI:
        worsen("DECLINE", f"DTI {dti:.1%} exceeds {config.MAX_DTI:.0%}", "LEND-1.2")
    elif dti > 0.36 and score < 700:
        worsen("REFER", f"DTI {dti:.1%} in 36-43% band with score < 700", "LEND-1.3")
    if score < config.MIN_CREDIT_SCORE:
        worsen("DECLINE", f"Credit score {score} below {config.MIN_CREDIT_SCORE}", "LEND-2.4")
    elif score < config.REFER_SCORE_BELOW:
        worsen("REFER", f"Credit score {score} in refer band", "LEND-2.3")
    elif score < 700 and dti > 0.36:
        worsen("REFER", f"Score {score} with DTI {dti:.1%} > 36%", "LEND-2.2")
    if hist < config.MIN_HISTORY_YEARS:
        worsen("REFER", f"Thin file: {hist} year(s) of credit history", "LEND-3.1")
    if ltv is not None:
        if ltv > 1.0:
            worsen("DECLINE", f"LTV {ltv:.0%} above 100%", "LEND-4.2")
        elif ltv > config.MAX_LTV:
            worsen("REFER", f"LTV {ltv:.0%} above {config.MAX_LTV:.0%}", "LEND-4.1")
    return {
        "estimated_payment": round(pay, 2), "dti": round(dti, 4),
        "ltv": round(ltv, 4) if ltv is not None else None,
        "rule_outcome": outcome, "rule_reasons": reasons,
    }


def make_nodes(llm, kg: KnowledgeGraph, retriever):
    intake_llm = llm.with_structured_output(IntakeResult)
    kyc_llm = llm.with_structured_output(KycAssessment)
    decision_llm = llm.with_structured_output(DecisionResult)

    # 1 ---------------------------------------------------------------
    def intake_extract(state: LoanState) -> dict:
        app = state["application"]
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a loan intake analyst. Summarize the application neutrally and list any risk "
                       "indicators in the notes. Do not invent facts."),
            ("human", "Application:\n{app}"),
        ])
        res: IntakeResult = (prompt | intake_llm).invoke({"app": json.dumps(app, indent=2)})
        return {"intake": res.model_dump(),
                "audit_trail": _audit(state, "intake_extract", res.summary)}

    # 2 ---------------------------------------------------------------
    def graph_retrieve(state: LoanState) -> dict:
        app = state["application"]
        kg.ensure_person(app["person_id"], app["applicant_name"], app.get("dob"))
        findings = kg.all_findings(app["person_id"], app["applicant_name"], app.get("dob"))
        triples = kg.neighborhood_triples(app["person_id"], hops=2)
        return {
            "graph_findings": findings,
            "graph_context": "\n".join(triples),
            "audit_trail": _audit(state, "graph_retrieve",
                                  f"{len(triples)} triples in 2-hop neighborhood; {len(findings)} findings "
                                  f"(max severity {max_severity(findings)})"),
        }

    # 3 ---------------------------------------------------------------
    def policy_retrieve(state: LoanState) -> dict:
        # Graph-first: enrich the vector query with what the graph found.
        hints = "; ".join(f["clause_hint"] for f in state["graph_findings"]) or "standard identity verification"
        query = f"KYC risk tiering and lending policy for: {hints}. DTI, credit score, thin file, escalation."
        docs = retriever.invoke(query)
        chunks = [{"chunk_id": d.metadata["chunk_id"], "text": d.page_content} for d in docs]
        return {"policy_chunks": chunks,
                "audit_trail": _audit(state, "policy_retrieve",
                                      f"query hints: {hints}; chunks: {[c['chunk_id'] for c in chunks]}")}

    # 4 ---------------------------------------------------------------
    def kyc_triage(state: LoanState) -> dict:
        findings = state["graph_findings"]
        floor = max_severity(findings)  # hard rule floor from graph evidence
        policies = "\n\n".join(f"[{c['chunk_id']}]\n{c['text']}" for c in state["policy_chunks"])
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a KYC/AML triage analyst. Assign a risk tier (LOW/MEDIUM/HIGH) using ONLY the "
                       "graph evidence and policy excerpts provided. Quote graph triples verbatim and cite clause "
                       "IDs. Do not tip off the applicant in the rationale."),
            ("human", "Application: {app}\nIntake: {intake}\n\nGraph findings:\n{findings}\n\n"
                      "Neighborhood triples:\n{triples}\n\nPolicy excerpts:\n{policies}"),
        ])
        res: KycAssessment = (prompt | kyc_llm).invoke({
            "app": json.dumps(state["application"]), "intake": json.dumps(state["intake"]),
            "findings": json.dumps(findings, indent=2), "triples": state["graph_context"], "policies": policies,
        })
        # The LLM may raise risk but never lower it below the deterministic floor.
        final = max(res.risk, floor, key=SEVERITY_ORDER.get)
        kyc = {**res.model_dump(), "llm_risk": res.risk, "rule_floor": floor, "risk": final}
        out = {"kyc": kyc, "audit_trail": _audit(state, "kyc_triage", f"risk={final} (llm={res.risk}, floor={floor})")}
        if final == "HIGH":
            out.update(decision="ESCALATE", needs_human=True,
                       rationale="HIGH KYC risk: escalated to Compliance. " + res.rationale,
                       citations={"graph": res.cited_graph_evidence, "policy": res.cited_policy_clauses})
        return out

    # 5 ---------------------------------------------------------------
    def credit_assess(state: LoanState) -> dict:
        m = credit_metrics(state["application"])
        return {"credit": m, "audit_trail": _audit(state, "credit_assess",
                f"DTI={m['dti']:.1%} LTV={m['ltv']} -> {m['rule_outcome']}: {m['rule_reasons']}")}

    # 6 ---------------------------------------------------------------
    def decision(state: LoanState) -> dict:
        policies = "\n\n".join(f"[{c['chunk_id']}]\n{c['text']}" for c in state["policy_chunks"])
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a senior loan underwriter. Recommend APPROVE, REFER or DECLINE and explain it "
                       "using the credit metrics, KYC assessment and policy excerpts. Follow the policy's "
                       "rule outcome unless a clear policy clause supports otherwise. Cite clause IDs."),
            ("human", "Application: {app}\nCredit metrics: {credit}\nKYC: {kyc}\n\nPolicy excerpts:\n{policies}"),
        ])
        res: DecisionResult = (prompt | decision_llm).invoke({
            "app": json.dumps(state["application"]), "credit": json.dumps(state["credit"]),
            "kyc": json.dumps(state["kyc"]), "policies": policies,
        })
        # Hard rules win: the final outcome can only be as lenient as the rules allow.
        order = {"APPROVE": 0, "REFER": 1, "DECLINE": 2}
        final = max(res.decision, state["credit"]["rule_outcome"], key=order.get)
        if final == "APPROVE" and state["kyc"]["risk"] == "MEDIUM":
            final = "REFER"  # KYC-2.2 / KYC-4.2: EDD cases are never auto-approved
        return {
            "decision": final, "llm_decision": res.decision,
            "rationale": res.rationale, "conditions": res.conditions,
            "needs_human": final == "REFER",
            "citations": {"policy": res.cited_policy_clauses, "graph": state["kyc"]["cited_graph_evidence"]},
            "audit_trail": _audit(state, "decision", f"llm={res.decision} rules={state['credit']['rule_outcome']} "
                                                      f"kyc={state['kyc']['risk']} -> {final}"),
        }

    # 7 ---------------------------------------------------------------
    def human_review(state: LoanState) -> dict:
        queue = "Compliance" if state["decision"] == "ESCALATE" else "Manual Underwriting"
        reviewer = interrupt({
            "queue": queue, "app_id": state["application"]["app_id"],
            "proposed_decision": state["decision"], "rationale": state["rationale"],
        })
        return {"human_review": {"queue": queue, **reviewer},
                "audit_trail": _audit(state, "human_review", f"{queue}: {reviewer}")}

    # 8 ---------------------------------------------------------------
    def finalize(state: LoanState) -> dict:
        out = {}
        action = (state.get("human_review") or {}).get("action", "")
        if action == "OVERRIDE_APPROVE":
            out["decision"] = "APPROVE"
        elif action == "OVERRIDE_DECLINE":
            out["decision"] = "DECLINE"
        final = out.get("decision", state["decision"])
        out["audit_trail"] = _audit(state, "finalize", f"final decision {final}")
        return out

    return dict(intake_extract=intake_extract, graph_retrieve=graph_retrieve, policy_retrieve=policy_retrieve,
                kyc_triage=kyc_triage, credit_assess=credit_assess, decision=decision,
                human_review=human_review, finalize=finalize)
