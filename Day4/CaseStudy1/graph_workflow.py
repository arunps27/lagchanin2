"""LangGraph wiring for the loan/KYC workflow."""
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from kg import KnowledgeGraph
from nodes import make_nodes
from state import LoanState


def build_workflow(llm, kg: KnowledgeGraph | None = None, retriever=None):
    if retriever is None:
        from vectorstore import get_retriever
        retriever = get_retriever()
    n = make_nodes(llm, kg or KnowledgeGraph(), retriever)

    g = StateGraph(LoanState)
    for name, fn in n.items():
        g.add_node(name, fn)

    g.add_edge(START, "intake_extract")
    g.add_edge("intake_extract", "graph_retrieve")
    g.add_edge("graph_retrieve", "policy_retrieve")
    g.add_edge("policy_retrieve", "kyc_triage")
    g.add_conditional_edges(
        "kyc_triage",
        lambda s: "human_review" if s["kyc"]["risk"] == "HIGH" else "credit_assess",
        {"human_review": "human_review", "credit_assess": "credit_assess"},
    )
    g.add_edge("credit_assess", "decision")
    g.add_conditional_edges(
        "decision",
        lambda s: "human_review" if s["decision"] == "REFER" else "finalize",
        {"human_review": "human_review", "finalize": "finalize"},
    )
    g.add_edge("human_review", "finalize")
    g.add_edge("finalize", END)
    return g.compile(checkpointer=MemorySaver())
