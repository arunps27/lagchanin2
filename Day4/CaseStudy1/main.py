"""CLI: run loan applications through the Graph RAG workflow.

  python main.py                 # all sample applications
  python main.py --app APP-004   # one application
  python main.py --interactive   # prompt for human-review decisions
  python main.py --mermaid       # print workflow diagram (Mermaid) and exit
"""
import argparse
import json

from langgraph.types import Command

import config
from graph_workflow import build_workflow


def run_app(wf, app: dict, interactive: bool) -> dict:
    cfg = {"configurable": {"thread_id": app["app_id"]}}
    wf.invoke({"application": app}, cfg)
    while wf.get_state(cfg).next:  # paused at human_review interrupt
        payload = wf.get_state(cfg).tasks[0].interrupts[0].value
        print(f"   ** PAUSED for {payload['queue']}: proposed {payload['proposed_decision']}")
        if interactive:
            action = input("   reviewer action [CONFIRM/OVERRIDE_APPROVE/OVERRIDE_DECLINE]: ").strip() or "CONFIRM"
            resume = {"reviewer": "human", "action": action}
        else:
            resume = {"reviewer": "auto-demo", "action": "CONFIRM"}
        wf.invoke(Command(resume=resume), cfg)
    return wf.get_state(cfg).values


def show(s: dict):
    app = s["application"]
    print(f"\n{'=' * 78}\n{app['app_id']}  {app['applicant_name']}  (loan {app['loan_amount']:,})")
    print(f"  KYC risk : {s['kyc']['risk']}  (llm={s['kyc']['llm_risk']}, rule floor={s['kyc']['rule_floor']})")
    if "credit" in s:
        c = s["credit"]
        print(f"  Credit   : DTI={c['dti']:.1%} LTV={c['ltv']} payment={c['estimated_payment']} rules->{c['rule_outcome']}")
    print(f"  DECISION : {s['decision']}")
    print(f"  Rationale: {s['rationale']}")
    cit = s.get("citations", {})
    if cit.get("policy"):
        print(f"  Policy   : {', '.join(cit['policy'])}")
    for e in cit.get("graph", [])[:6]:
        print(f"  Graph    : {e}")
    if s.get("human_review"):
        print(f"  Human    : {s['human_review']}")
    print("  Audit    : " + " | ".join(a["step"] for a in s["audit_trail"]))


def main():
    import sys
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # Windows cp1252 consoles
    p = argparse.ArgumentParser()
    p.add_argument("--app")
    p.add_argument("--interactive", action="store_true")
    p.add_argument("--mermaid", action="store_true")
    p.add_argument("--json", action="store_true", help="dump full final state as JSON")
    args = p.parse_args()

    from kg import KnowledgeGraph
    llm = None if args.mermaid else config.get_llm()
    wf = build_workflow(llm, KnowledgeGraph()) if not args.mermaid else build_workflow(_NullLLM(), KnowledgeGraph())
    if args.mermaid:
        print(wf.get_graph().draw_mermaid())
        return

    apps = json.loads((config.DATA_DIR / "applications.json").read_text(encoding="utf-8"))
    if args.app:
        apps = [a for a in apps if a["app_id"] == args.app]
        if not apps:
            raise SystemExit(f"Unknown application {args.app}")
    for app in apps:
        state = run_app(wf, app, args.interactive)
        print(json.dumps(state, indent=2, default=str)) if args.json else show(state)


class _NullLLM:
    """Stand-in so the graph can be built for diagram export without an API key."""
    def with_structured_output(self, *_a, **_k):
        return self


if __name__ == "__main__":
    main()
