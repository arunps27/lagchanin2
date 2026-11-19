"""LangGraph Graph-RAG workflow: Automated Loan Processing & KYC Triage.

Plugin : LangChain community `PyPDFLoader` (open-source document-loader plugin) -> KYC PDF extraction
MCP    : official open-source `mcp-server-sqlite` (stdio) -> bank records (customers, watchlist, policy...)
Graph  : networkx knowledge graph built from MCP rows; multi-hop subgraph retrieval feeds the LLM (Graph RAG)
LLM    : Groq `openai/gpt-oss-120b`
"""
import ast, json, os, re, sys, difflib
from datetime import date
from pathlib import Path
from typing import TypedDict, Any
import networkx as nx
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END
from langgraph.config import get_stream_writer
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")
DB = ROOT / "data" / "bank.db"
DOCS = ROOT / "data" / "docs"
MODEL = "openai/gpt-oss-120b"


class State(TypedDict, total=False):
    app_id: str
    doc_text: str
    doc_fields: dict
    rows: dict
    graph: Any
    subgraph: dict
    signals: dict
    triage: dict
    underwriting: dict
    decision: dict


def llm_json(system: str, user: str) -> dict:
    llm = ChatGroq(model=MODEL, temperature=0, api_key=os.environ.get("GROQ_API_KEY"))
    msg = llm.invoke([("system", system + "\nReply with ONLY one valid JSON object."), ("user", user)])
    m = re.search(r"\{.*\}", msg.content, re.S)
    return json.loads(m.group(0))


# ---------------- node 1: PLUGIN (PyPDFLoader) ----------------
def extract_docs(state: State):
    w = get_stream_writer()
    path = DOCS / f"{state['app_id']}_kyc.pdf"
    w({"type": "plugin_start", "plugin": "langchain_community.document_loaders.PyPDFLoader", "input": path.name})
    pages = PyPDFLoader(str(path)).load()
    text = "\n".join(p.page_content for p in pages)
    pats = {"name": r"Full Name:\s*(.+)", "dob": r"Date of Birth:\s*(.+)", "address": r"Address:\s*(.+)",
            "phone": r"Phone:\s*(.+)", "id_number": r"Government ID:\s*(\S+)", "id_expiry": r"expires (\d{4}-\d{2}-\d{2})",
            "employer": r"Employer:\s*(.+)", "doc_income": r"INR\s*(\d+)"}
    f = {}
    for k, pat in pats.items():
        mm = re.search(pat, text)
        f[k] = mm.group(1).strip() if mm else None
    w({"type": "plugin_end", "pages": len(pages), "text": text, "fields": f})
    return {"doc_text": text, "doc_fields": f}


# ---------------- node 2: MCP (sqlite server) ----------------
async def fetch_via_mcp(state: State):
    w = get_stream_writer()
    exe = Path(sys.executable).parent / ("mcp-server-sqlite.exe" if os.name == "nt" else "mcp-server-sqlite")
    params = StdioServerParameters(command=str(exe), args=["--db-path", str(DB)])
    rows = {}
    async with stdio_client(params) as (r, wr):
        async with ClientSession(r, wr) as s:
            await s.initialize()
            tools = await s.list_tools()
            w({"type": "mcp_server", "server": "mcp-server-sqlite (modelcontextprotocol/servers)", "tools": [t.name for t in tools.tools]})
            for tbl in ["customers", "applications", "accounts", "watchlist", "policy"]:
                q = f"SELECT * FROM {tbl}"
                w({"type": "mcp_call", "tool": "read_query", "args": {"query": q}})
                res = await s.call_tool("read_query", {"query": q})
                txt = res.content[0].text
                rows[tbl] = ast.literal_eval(txt)
                w({"type": "mcp_result", "tool": "read_query", "table": tbl, "count": len(rows[tbl]), "preview": txt[:400]})
    return {"rows": rows}


# ---------------- node 3: build knowledge graph ----------------
def build_graph(state: State):
    w = get_stream_writer()
    R = state["rows"]
    G = nx.Graph()
    rel = {"Address": "LIVES_AT", "Phone": "HAS_PHONE", "Employer": "WORKS_AT"}
    for c in R["customers"]:
        G.add_node(f"cust:{c['id']}", label=c["name"], kind="Customer")
        for kind, key in (("Address", "address"), ("Phone", "phone"), ("Employer", "employer")):
            n = f"{kind.lower()}:{c[key]}"
            G.add_node(n, label=c[key], kind=kind)
            G.add_edge(f"cust:{c['id']}", n, rel=rel[kind])
    for a in R["accounts"]:
        n = f"acct:{a['id']}"
        G.add_node(n, label=f"{a['bank']} ({a['balance']:,})", kind="Account")
        G.add_edge(f"cust:{a['customer_id']}", n, rel="OWNS")
    for a in R["applications"]:
        n = f"app:{a['app_id']}"
        G.add_node(n, label=a["app_id"], kind="Application")
        G.add_edge(f"cust:{a['customer_id']}", n, rel="APPLIED")
    for wl in R["watchlist"]:
        G.add_node(f"wl:{wl['wid']}", label=f"{wl['name']} [{wl['source']}]", kind="Watchlist", note=wl["note"], source=wl["source"])
    # entity-resolution edges: customers / employers ~ watchlist (fuzzy name match)
    for n, d in list(G.nodes(data=True)):
        if d["kind"] in ("Customer", "Employer"):
            for wl in R["watchlist"]:
                sc = difflib.SequenceMatcher(None, d["label"].lower(), wl["name"].lower()).ratio()
                if sc >= 0.85:
                    G.add_edge(n, f"wl:{wl['wid']}", rel="MATCHES_WATCHLIST", score=round(sc, 2))
    w({"type": "graph_built", "nodes": G.number_of_nodes(), "edges": G.number_of_edges()})
    return {"graph": G}


# ---------------- node 4: graph retrieval (multi-hop subgraph) ----------------
def graph_retrieve(state: State):
    w = get_stream_writer()
    G: nx.Graph = state["graph"]
    R = state["rows"]
    app = next(a for a in R["applications"] if a["app_id"] == state["app_id"])
    cust = next(c for c in R["customers"] if c["id"] == app["customer_id"])
    center = f"cust:{cust['id']}"

    keep = {center}
    frontier = {center}
    for _ in range(3):  # 3 hops: customer -> attribute -> other customer -> their attributes/watchlist
        nxt = set()
        for n in frontier:
            nxt.update(G.neighbors(n))
        keep |= nxt
        frontier = nxt
    S = G.subgraph(keep)

    hits, ring = [], []
    def wl_hits(node):
        out = []
        for nb in G.neighbors(node):
            if G[node][nb].get("rel") == "MATCHES_WATCHLIST":
                out.append({"entity": G.nodes[node]["label"], "list": G.nodes[nb]["label"], "source": G.nodes[nb]["source"],
                            "note": G.nodes[nb]["note"], "score": G[node][nb]["score"]})
        return out
    hits += wl_hits(center)
    for nb in G.neighbors(center):
        k = G.nodes[nb]["kind"]
        if k in ("Address", "Phone", "Employer"):
            for h in wl_hits(nb):
                h["entity"] = f"{h['entity']} (employer)"; hits.append(h)
            for other in G.neighbors(nb):
                if other != center and other.startswith("cust:"):
                    ring.append({"customer": G.nodes[other]["label"], "via": k, "value": G.nodes[nb]["label"]})
    doc_inc = int(state["doc_fields"].get("doc_income") or 0)
    sig = {
        "watchlist_hits": hits,
        "ring_members": ring,
        "kyc_expired": cust["id_expiry"] < date.today().isoformat(),
        "income_mismatch": abs(cust["stated_income"] - doc_inc) > 0.2 * max(doc_inc, 1),
        "income_stated_vs_document": [cust["stated_income"], doc_inc],
    }
    nodes = [{"id": n, "label": d["label"], "kind": d["kind"], "center": n == center} for n, d in S.nodes(data=True)]
    edges = [{"from": u, "to": v, "label": d["rel"]} for u, v, d in S.edges(data=True)]
    w({"type": "graph_subgraph", "nodes": nodes, "edges": edges, "signals": sig})
    return {"subgraph": {"nodes": nodes, "edges": edges, "customer": cust, "application": app, "policy": R["policy"]}, "signals": sig}


# ---------------- node 5: LLM KYC triage ----------------
def kyc_triage(state: State):
    w = get_stream_writer()
    sg = state["subgraph"]
    labels = {n["id"]: n["label"] for n in sg["nodes"]}
    ctx = "\n".join(f"{labels[e['from']]} -[{e['label']}]-> {labels[e['to']]}" for e in sg["edges"])
    prompt = (f"Applicant record: {json.dumps(sg['customer'])}\nDocument fields (from PDF plugin): {json.dumps(state['doc_fields'])}\n"
              f"Graph context (retrieved subgraph triples):\n{ctx}\nGraph-derived signals: {json.dumps(state['signals'])}\n"
              "Bank policy:\n" + "\n".join(f"{p['pid']} {p['topic']}: {p['rule']}" for p in sg["policy"]) +
              '\nReturn JSON {"kyc_status":"PASS|FAIL","risk_level":"LOW|MEDIUM|HIGH|CRITICAL","flags":[strings],'
              '"triage_route":"UNDERWRITE|MANUAL_REVIEW|REJECT_COMPLIANCE","reasoning":"short, cite policy ids"}')
    w({"type": "llm_start", "step": "kyc_triage", "model": MODEL})
    out = llm_json("You are a KYC/AML triage analyst at a bank. Use only the provided graph context and policy.", prompt)
    w({"type": "llm_end", "step": "kyc_triage", "output": out})
    return {"triage": out}


def route(state: State) -> str:
    s, t = state["signals"], state["triage"]
    if s["watchlist_hits"] or t.get("triage_route") == "REJECT_COMPLIANCE":
        return "compliance_reject"
    if s["ring_members"] or s["kyc_expired"] or s["income_mismatch"] or t.get("triage_route") == "MANUAL_REVIEW" or t.get("kyc_status") == "FAIL":
        return "manual_review"
    return "underwrite"


# ---------------- terminal nodes ----------------
def emi(p, months, rate=0.11):
    r = rate / 12
    return round(p * r * (1 + r) ** months / ((1 + r) ** months - 1))


def underwrite(state: State):
    w = get_stream_writer()
    sg = state["subgraph"]
    a, c = sg["application"], sg["customer"]
    e = emi(a["amount"], a["tenure_months"])
    inc_m = c["stated_income"] / 12
    dti = (a["monthly_obligations"] + e) / inc_m
    uw = {"emi": e, "monthly_income": round(inc_m), "dti": round(dti, 3)}
    w({"type": "node_info", "step": "underwrite", "data": uw})
    dec = "AUTO_APPROVE" if dti <= 0.40 else ("REJECT" if dti > 0.50 else "MANUAL_REVIEW")
    narrative = llm_json("You are a credit officer. Write a 2-3 sentence customer-ready rationale.",
                         f'Decision {dec}; underwriting {json.dumps(uw)}; policies P-4 and P-6. Return {{"narrative":"..."}}')["narrative"]
    d = {"decision": dec, "reasons": [f"DTI {dti:.1%} vs policy threshold (P-4/P-6)"], "narrative": narrative}
    w({"type": "llm_end", "step": "underwrite", "output": d})
    return {"underwriting": uw, "decision": d}


def manual_review(state: State):
    s = state["signals"]
    reasons = [r for r, on in (("Expired government ID (P-1)", s["kyc_expired"]),
                               ("Stated income differs from document by >20% (P-5)", s["income_mismatch"]),
                               ("Shared address/phone/employer cluster (P-3)", bool(s["ring_members"]))) if on]
    reasons += [f"Triage flag: {f}" for f in state["triage"].get("flags", [])][:3]
    narrative = llm_json("You are a KYC ops lead. Write 2 sentences summarising why this goes to manual review.",
                         f'Reasons {reasons}. Return {{"narrative":"..."}}')["narrative"]
    d = {"decision": "MANUAL_REVIEW", "queue": "Fraud / KYC ops", "reasons": reasons, "narrative": narrative}
    get_stream_writer()({"type": "llm_end", "step": "manual_review", "output": d})
    return {"decision": d}


def compliance_reject(state: State):
    s = state["signals"]
    reasons = [f"Watchlist match: {h['entity']} ~ {h['list']} (score {h['score']})" for h in s["watchlist_hits"]] or ["Triage flagged compliance risk (P-2)"]
    narrative = llm_json("You are a compliance officer. Write 2 sentences, internal use.",
                         f'Watchlist hits {json.dumps(s["watchlist_hits"])}; policy P-2. Return {{"narrative":"..."}}')["narrative"]
    d = {"decision": "REJECT_COMPLIANCE", "queue": "Compliance escalation", "reasons": reasons, "narrative": narrative}
    get_stream_writer()({"type": "llm_end", "step": "compliance_reject", "output": d})
    return {"decision": d}


def build_app():
    g = StateGraph(State)
    for name, fn in [("extract_docs", extract_docs), ("fetch_via_mcp", fetch_via_mcp), ("build_graph", build_graph),
                     ("graph_retrieve", graph_retrieve), ("kyc_triage", kyc_triage), ("underwrite", underwrite),
                     ("manual_review", manual_review), ("compliance_reject", compliance_reject)]:
        g.add_node(name, fn)
    g.add_edge(START, "extract_docs")
    g.add_edge("extract_docs", "fetch_via_mcp")
    g.add_edge("fetch_via_mcp", "build_graph")
    g.add_edge("build_graph", "graph_retrieve")
    g.add_edge("graph_retrieve", "kyc_triage")
    g.add_conditional_edges("kyc_triage", route, {"underwrite": "underwrite", "manual_review": "manual_review", "compliance_reject": "compliance_reject"})
    for n in ("underwrite", "manual_review", "compliance_reject"):
        g.add_edge(n, END)
    return g.compile()
