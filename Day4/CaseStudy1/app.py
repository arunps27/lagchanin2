"""Streamlit UI: upload policy docs / applications, run the Graph RAG workflow, view KPIs.

  streamlit run app.py
"""
import io
import json
import time
import uuid
from datetime import datetime

import pandas as pd
import streamlit as st
from langgraph.types import Command

import config
from graph_workflow import build_workflow
from kg import KnowledgeGraph
from vectorstore import get_retriever, get_store

SAMPLE_DIR = config.BASE_DIR / "sample_uploads"
RESULTS_FILE = config.BASE_DIR / "results.jsonl"
REQUIRED = ["app_id", "applicant_name", "monthly_income", "existing_monthly_debt",
            "loan_amount", "loan_term_months", "credit_score", "credit_history_years"]
NUMERIC = REQUIRED[2:]

st.set_page_config(page_title="Loan & KYC Graph RAG", page_icon=":bank:", layout="wide")


# ---------------- resources & persistence ----------------
@st.cache_resource(show_spinner="Loading vector store...")
def cached_store():
    return get_store()


@st.cache_resource(show_spinner="Loading models, graph and vector store...")
def get_wf():
    return build_workflow(config.get_llm(), KnowledgeGraph(), get_retriever(store=cached_store()))


@st.cache_resource
def test_kg():
    return KnowledgeGraph()


def reindex(file_map: dict[str, bytes]):
    """Save files into the policy folder, rebuild the vector store, reload cached resources."""
    import ingest
    from contextlib import redirect_stdout
    for name, data in file_map.items():
        (config.POLICY_DIR / name).write_bytes(data)
    buf = io.StringIO()
    with st.spinner("Ingesting..."), redirect_stdout(buf):
        ingest.main()
    get_wf.clear()
    cached_store.clear()
    return buf.getvalue()


def load_records() -> list[dict]:
    if not RESULTS_FILE.exists():
        return []
    return [json.loads(l) for l in RESULTS_FILE.read_text(encoding="utf-8").splitlines() if l.strip()]


def save_record(rec: dict):
    with RESULTS_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")


ss = st.session_state
ss.setdefault("apps", [])        # queued applications
ss.setdefault("pending", {})     # thread_id -> {cfg, app, elapsed, state}
ss.setdefault("last", {})        # app_id -> final state (for detail view)


def to_record(state: dict, elapsed: float, human: bool) -> dict:
    app, kyc, credit = state["application"], state["kyc"], state.get("credit") or {}
    cit = state.get("citations", {})
    return {
        "ts": datetime.now().isoformat(timespec="seconds"),
        "app_id": app["app_id"], "name": app["applicant_name"], "loan_amount": app["loan_amount"],
        "decision": state["decision"], "llm_decision": state.get("llm_decision"),
        "kyc_risk": kyc["risk"], "llm_kyc_risk": kyc["llm_risk"], "rule_floor": kyc["rule_floor"],
        "dti": credit.get("dti"), "credit_outcome": credit.get("rule_outcome"),
        "human_touch": human, "policy_cites": len(cit.get("policy", [])),
        "graph_cites": len(cit.get("graph", [])), "latency_s": round(elapsed, 2),
        "rationale": state.get("rationale", ""),
    }


def finish(state: dict, elapsed: float, human: bool):
    ss.last[state["application"]["app_id"]] = state
    save_record(to_record(state, elapsed, human))


def run_one(app: dict):
    wf = get_wf()
    tid = f"{app['app_id']}-{uuid.uuid4().hex[:6]}"
    cfg = {"configurable": {"thread_id": tid}}
    t0 = time.time()
    wf.invoke({"application": app}, cfg)
    elapsed = time.time() - t0
    snap = wf.get_state(cfg)
    if snap.next:  # paused at human_review
        ss.pending[tid] = {"cfg": cfg, "app": app, "elapsed": elapsed,
                           "payload": snap.tasks[0].interrupts[0].value, "state": snap.values}
    else:
        finish(snap.values, elapsed, human=False)


def resume(tid: str, action: str):
    wf, p = get_wf(), ss.pending.pop(tid)
    t0 = time.time()
    wf.invoke(Command(resume={"reviewer": "ui-user", "action": action}), p["cfg"])
    finish(wf.get_state(p["cfg"]).values, p["elapsed"] + time.time() - t0, human=True)


def normalize_apps(df: pd.DataFrame) -> list[dict]:
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns/fields: {missing}")
    out = []
    for r in df.where(pd.notna(df), None).to_dict("records"):
        r["app_id"] = str(r["app_id"])
        for c in NUMERIC:
            r[c] = float(r[c]) if c in ("monthly_income", "existing_monthly_debt", "loan_amount") else int(r[c])
        r.setdefault("person_id", None)
        r["person_id"] = r.get("person_id") or f"NEW-{r['app_id']}"
        r["notes"] = r.get("notes") or ""
        out.append(r)
    return out


# ---------------- sidebar ----------------
with st.sidebar:
    st.title("Loan & KYC Graph RAG")
    st.caption(f"Model: `{config.GROQ_MODEL}` via Groq")
    st.write(f"Policy files: **{len([p for p in config.POLICY_DIR.iterdir() if p.is_file()])}**")
    if st.button("Clear KPI history"):
        RESULTS_FILE.unlink(missing_ok=True)
        ss.last.clear()
        st.rerun()

tab_docs, tab_apps, tab_review, tab_kpi, tab_test = st.tabs(
    ["1. Policy documents", "2. Applications", "3. Human review", "4. KPIs", "5. Test lab"])

# ---------------- tab 1: policy docs ----------------
with tab_docs:
    st.subheader("Upload policy / KYC documents")
    st.caption("Uploaded files are saved to data/policies, then re-run through "
               "Corpus → Chunk → Tokenization → Embedding → Indexing → Vector DB Store.")
    files = st.file_uploader("PDF, Markdown or text", type=["pdf", "md", "txt"], accept_multiple_files=True)
    if st.button("Upload & re-index", type="primary", disabled=not files):
        log = reindex({f.name: f.getvalue() for f in files})
        st.success(f"Indexed {len(files)} new file(s).")
        st.code(log, language="text")
    sample_pdfs = sorted(SAMPLE_DIR.glob("*.pdf"))
    if sample_pdfs and st.button(f"Quick-load the {len(sample_pdfs)} sample PDFs (20/40/60/80 pages)"):
        log = reindex({p.name: p.read_bytes() for p in sample_pdfs})
        st.success("Sample PDFs indexed.")
        st.code(log, language="text")
    st.markdown("**Current corpus**")
    st.dataframe(pd.DataFrame(
        [{"file": p.name, "size_kb": round(p.stat().st_size / 1024, 1)}
         for p in sorted(config.POLICY_DIR.iterdir()) if p.is_file()]), hide_index=True)

# ---------------- tab 2: applications ----------------
with tab_apps:
    st.subheader("Add applications")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("Load 6 sample applications"):
            ss.apps = json.loads((config.DATA_DIR / "applications.json").read_text(encoding="utf-8"))
        for sf in sorted(SAMPLE_DIR.glob("applications_batch_*")):
            if st.button(f"Quick-add {sf.name}"):
                df = pd.read_csv(sf) if sf.suffix == ".csv" else pd.DataFrame(json.loads(sf.read_text(encoding="utf-8")))
                ss.apps = ss.apps + normalize_apps(df)
                st.rerun()
        up = st.file_uploader("Upload applications (JSON or CSV)", type=["json", "csv"], key="appup")
        if up and st.button("Add uploaded applications"):
            try:
                if up.name.endswith(".csv"):
                    df = pd.read_csv(up)
                else:
                    data = json.load(up)
                    df = pd.DataFrame(data if isinstance(data, list) else [data])
                ss.apps = ss.apps + normalize_apps(df)
                st.success(f"Added {len(df)} application(s).")
            except Exception as e:
                st.error(str(e))
    with c2:
        with st.form("manual"):
            st.markdown("**Manual entry**")
            a, b = st.columns(2)
            app_id = a.text_input("App ID", f"APP-{len(ss.apps) + 101}")
            name = b.text_input("Applicant name")
            dob = a.text_input("DOB (YYYY-MM-DD)")
            income = b.number_input("Monthly income", 0.0, value=6000.0)
            debt = a.number_input("Existing monthly debt", 0.0, value=500.0)
            loan = b.number_input("Loan amount", 0.0, value=50000.0)
            term = a.number_input("Term (months)", 1, value=60)
            score = b.number_input("Credit score", 300, 850, 700)
            hist = a.number_input("Credit history (years)", 0, value=5)
            pv = b.number_input("Property value (0 = unsecured)", 0.0, value=0.0)
            notes = st.text_area("Notes")
            if st.form_submit_button("Add") and name:
                ss.apps.append({"app_id": app_id, "person_id": f"NEW-{app_id}", "applicant_name": name,
                                "dob": dob or None, "monthly_income": income, "existing_monthly_debt": debt,
                                "loan_amount": loan, "loan_term_months": int(term), "credit_score": int(score),
                                "credit_history_years": int(hist), "property_value": pv or None, "notes": notes})
                st.rerun()

    st.divider()
    st.markdown(f"**Queue: {len(ss.apps)} application(s)**")
    if ss.apps:
        st.dataframe(pd.DataFrame(ss.apps)[["app_id", "applicant_name", "loan_amount", "monthly_income",
                                            "credit_score", "credit_history_years"]], hide_index=True)
        if st.button("Run workflow on all", type="primary"):
            bar = st.progress(0.0)
            for i, app in enumerate(ss.apps):
                try:
                    run_one(app)
                except Exception as e:
                    st.error(f"{app['app_id']} failed: {e}")
                bar.progress((i + 1) / len(ss.apps), text=f"{app['app_id']} done")
            ss.apps = []
            st.success("Done. See 'Human review' for paused cases and 'KPIs' for performance.")

    if ss.last:
        st.markdown("#### Results")
        for aid, s in ss.last.items():
            icon = {"APPROVE": "✅", "DECLINE": "⛔", "REFER": "🟡", "ESCALATE": "🚨"}.get(s["decision"], "")
            with st.expander(f"{icon} {aid} · {s['application']['applicant_name']} · {s['decision']} "
                             f"· KYC {s['kyc']['risk']}"):
                st.write(s["rationale"])
                cit = s.get("citations", {})
                st.caption("Policy clauses: " + (", ".join(cit.get("policy", [])) or "-"))
                if cit.get("graph"):
                    st.code("\n".join(cit["graph"]), language="text")
                if s.get("credit"):
                    st.json(s["credit"])
                st.caption("Audit: " + " → ".join(a["step"] for a in s["audit_trail"]))

# ---------------- tab 3: human review ----------------
with tab_review:
    st.subheader(f"Pending human review ({len(ss.pending)})")
    if not ss.pending:
        st.info("No cases waiting.")
    for tid, p in list(ss.pending.items()):
        pl = p["payload"]
        with st.container(border=True):
            st.markdown(f"**{pl['app_id']}** · queue: **{pl['queue']}** · proposed: **{pl['proposed_decision']}**")
            st.write(pl["rationale"])
            b1, b2, b3 = st.columns(3)
            if b1.button("Confirm", key=f"c{tid}"):
                resume(tid, "CONFIRM"); st.rerun()
            if b2.button("Override: Approve", key=f"a{tid}"):
                resume(tid, "OVERRIDE_APPROVE"); st.rerun()
            if b3.button("Override: Decline", key=f"d{tid}"):
                resume(tid, "OVERRIDE_DECLINE"); st.rerun()

# ---------------- tab 4: KPIs ----------------
with tab_kpi:
    recs = load_records()
    pending_n = len(ss.pending)
    if not recs:
        st.info("No processed applications yet. Run some from the Applications tab.")
    else:
        df = pd.DataFrame(recs)
        n = len(df)
        stp = (~df.human_touch).mean()
        approved = df[df.decision == "APPROVE"]
        # Guardrail interventions: places where deterministic rules changed the LLM output.
        overrides = ((df.llm_kyc_risk != df.kyc_risk) | (df.llm_decision.notna() & (df.llm_decision != df.decision)
                                                         & ~df.human_touch)).mean()
        cited = ((df.policy_cites > 0) & (df.graph_cites + (df.kyc_risk == "LOW") > 0)).mean()

        r1 = st.columns(5)
        r1[0].metric("Applications processed", n)
        r1[1].metric("Straight-through rate", f"{stp:.0%}", help="Decided with no human touch")
        r1[2].metric("Approval rate", f"{(df.decision == 'APPROVE').mean():.0%}")
        r1[3].metric("Escalation rate (KYC)", f"{(df.decision == 'ESCALATE').mean():.0%}")
        r1[4].metric("Referral rate", f"{(df.decision == 'REFER').mean():.0%}")
        r2 = st.columns(5)
        r2[0].metric("Avg latency / app", f"{df.latency_s.mean():.1f}s")
        r2[1].metric("P95 latency", f"{df.latency_s.quantile(0.95):.1f}s")
        r2[2].metric("Approved volume", f"${approved.loan_amount.sum():,.0f}")
        r2[3].metric("Avg DTI (credit-assessed)", f"{df.dti.mean():.1%}" if df.dti.notna().any() else "-")
        r2[4].metric("Pending human review", pending_n)
        r3 = st.columns(3)
        r3[0].metric("Guardrail override rate", f"{overrides:.0%}",
                     help="Share of cases where hard rules changed the LLM's KYC tier or decision")
        r3[1].metric("Policy-cited decisions", f"{(df.policy_cites > 0).mean():.0%}",
                     help="Explainability: decisions citing at least one policy clause")
        r3[2].metric("Graph-evidenced decisions", f"{(df.graph_cites > 0).mean():.0%}",
                     help="Decisions citing graph relationships")

        c1, c2 = st.columns(2)
        c1.markdown("**Decisions**")
        c1.bar_chart(df.decision.value_counts())
        c2.markdown("**KYC risk tiers**")
        c2.bar_chart(df.kyc_risk.value_counts())
        c3, c4 = st.columns(2)
        c3.markdown("**Latency per application (s)**")
        c3.bar_chart(df.set_index("app_id").latency_s)
        c4.markdown("**Loan volume by decision ($)**")
        c4.bar_chart(df.groupby("decision").loan_amount.sum())

        st.markdown("**Decision log**")
        st.dataframe(df[["ts", "app_id", "name", "loan_amount", "decision", "kyc_risk", "dti",
                         "human_touch", "latency_s"]], hide_index=True)
        st.download_button("Download log (CSV)", df.to_csv(index=False), "decision_log.csv")

# ---------------- tab 5: test lab ----------------
with tab_test:
    st.subheader("Test lab")
    st.caption("Try each part of the Graph RAG in isolation before running full applications.")

    st.markdown("### A. Vector search over your policy documents")
    q = st.text_input("Ask a policy question", "What happens if an applicant shares a phone number with a fraud-flagged customer?")
    k = st.slider("Results", 1, 10, 4)
    if st.button("Search policies") and q.strip():
        hits = cached_store().similarity_search_with_score(q, k=k)
        for d, score in hits:
            with st.container(border=True):
                st.caption(f"{d.metadata['chunk_id']} · {d.metadata['source']} · cosine {float(score):.3f}")
                st.write(d.page_content)

    st.markdown("### B. Knowledge-graph screening (no LLM)")
    g1, g2, g3 = st.columns(3)
    t_name = g1.text_input("Name", "Karim Rahman")
    t_dob = g2.text_input("Date of birth", "1961-12-01")
    t_pid = g3.text_input("Person ID in graph (optional, e.g. P-004, P-008)", "")
    if st.button("Screen") and t_name.strip():
        kgr = test_kg()
        if t_pid and t_pid in kgr.g:
            findings = kgr.all_findings(t_pid, t_name, t_dob or None)
            triples = kgr.neighborhood_triples(t_pid, 2)
        else:
            findings = kgr.watchlist_findings(t_name, t_dob or None)
            triples = []
            if t_pid:
                st.warning(f"{t_pid} is not in the graph; showing watchlist screening only.")
        from kg import max_severity
        st.metric("Rule-based KYC floor", max_severity(findings))
        st.json(findings) if findings else st.success("No findings.")
        if triples:
            st.code("\n".join(triples), language="text")

    st.markdown("### C. Single application, end to end")
    st.caption("Runs the full LangGraph workflow (calls Groq). Results land in KPIs.")
    presets = {
        "Clean approve": {"applicant_name": "Test Clean", "monthly_income": 9000, "existing_monthly_debt": 500,
                          "loan_amount": 80000, "loan_term_months": 120, "credit_score": 770, "credit_history_years": 10},
        "High DTI (expect decline)": {"applicant_name": "Test HighDTI", "monthly_income": 3500, "existing_monthly_debt": 1800,
                                      "loan_amount": 40000, "loan_term_months": 60, "credit_score": 720, "credit_history_years": 8},
        "Sanctions name (expect escalate)": {"applicant_name": "Ivan Petrovic Sokolov", "dob": "1975-03-12", "monthly_income": 12000,
                                             "existing_monthly_debt": 800, "loan_amount": 100000, "loan_term_months": 120,
                                             "credit_score": 760, "credit_history_years": 15},
        "Fraud-ring phone (expect escalate)": {"applicant_name": "Elena Cruz", "dob": "1994-02-17", "person_id": "P-006",
                                              "monthly_income": 5200, "existing_monthly_debt": 450, "loan_amount": 30000,
                                              "loan_term_months": 60, "credit_score": 715, "credit_history_years": 5},
    }
    choice = st.selectbox("Preset", list(presets))
    if st.button("Run this test application"):
        base = {"notes": "", "property_value": None, "dob": None, **presets[choice]}
        n_ = len(ss.last) + len(ss.pending) + 1
        base["app_id"] = f"TEST-{n_:03d}"
        base["person_id"] = base.get("person_id") or f"NEW-{base['app_id']}"
        with st.spinner("Running workflow..."):
            run_one(base)
        if ss.pending:
            st.info("Paused for human review; handle it in the 'Human review' tab.")
        else:
            s = ss.last[base["app_id"]]
            st.success(f"Decision: {s['decision']} · KYC {s['kyc']['risk']}")
            st.write(s["rationale"])
