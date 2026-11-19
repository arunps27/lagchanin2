"""RAG chatbot: LangChain + LangGraph + Groq (openai/gpt-oss-120b) with a FastAPI UI."""
import io
import os
import re
import uuid
from typing import List, Optional, TypedDict

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel

load_dotenv()
MODEL = "openai/gpt-oss-120b"
BASE = os.path.dirname(os.path.abspath(__file__))

app = FastAPI(title="LangGraph RAG")

# ---------------------------------------------------------------- global state
S = {
    "api_key": os.getenv("GROQ_API_KEY", ""),
    "store": None,          # vector store (after upload)
    "files": [],
    # ordered pipeline; "core" nodes can't be removed
    "nodes": [
        {"id": "rewrite_query", "kind": "builtin", "core": False, "enabled": True,
         "desc": "LLM rewrites the question into a standalone search query"},
        {"id": "retrieve", "kind": "builtin", "core": True, "enabled": True,
         "desc": "Vector search over uploaded data"},
        {"id": "grade_documents", "kind": "builtin", "core": False, "enabled": False,
         "desc": "LLM drops retrieved chunks that are irrelevant"},
        {"id": "generate", "kind": "builtin", "core": True, "enabled": True,
         "desc": "LLM answers using the retrieved context"},
        {"id": "add_citations", "kind": "builtin", "core": False, "enabled": False,
         "desc": "Appends the source files to the answer"},
    ],
}
_emb = None
_graph = None


def embeddings():
    global _emb
    if _emb is None:
        _emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return _emb


def llm():
    if not S["api_key"]:
        raise HTTPException(400, "Groq API key not set. Enter it in the UI or set GROQ_API_KEY.")
    return ChatGroq(model=MODEL, api_key=S["api_key"], temperature=0.2)


def text_of(msg) -> str:
    c = msg.content
    return c if isinstance(c, str) else "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in c)


# ---------------------------------------------------------------- graph state/nodes
class State(TypedDict, total=False):
    question: str
    query: str
    docs: List[Document]
    answer: str


def n_rewrite_query(s: State):
    r = llm().invoke(
        "Rewrite the user question as a concise standalone search query. "
        f"Output only the query.\n\nQuestion: {s['question']}")
    return {"query": text_of(r).strip() or s["question"]}


def n_retrieve(s: State):
    docs = S["store"].similarity_search(s.get("query") or s["question"], k=5)
    return {"docs": docs}


def n_grade_documents(s: State):
    keep = []
    for d in s["docs"]:
        r = llm().invoke(
            f"Question: {s['question']}\n\nPassage:\n{d.page_content[:1500]}\n\n"
            "Is the passage relevant to the question? Answer only 'yes' or 'no'.")
        if "yes" in text_of(r).lower():
            keep.append(d)
    return {"docs": keep}


def n_generate(s: State):
    ctx = "\n\n---\n\n".join(d.page_content for d in s["docs"]) or "(no relevant context found)"
    r = llm().invoke(
        "You are a helpful assistant. Answer the question using ONLY the context below. "
        "If the answer is not in the context, say you don't know.\n\n"
        f"Context:\n{ctx}\n\nQuestion: {s['question']}")
    return {"answer": text_of(r).strip()}


def n_add_citations(s: State):
    srcs = sorted({d.metadata.get("source", "?") for d in s["docs"]})
    return {"answer": s["answer"] + ("\n\nSources: " + ", ".join(srcs) if srcs else "")}


def make_custom(instruction: str):
    def node(s: State):
        r = llm().invoke(
            f"Apply this instruction to the answer below. Output only the revised answer.\n"
            f"Instruction: {instruction}\n\nAnswer:\n{s['answer']}")
        return {"answer": text_of(r).strip()}
    return node


BUILTIN = {
    "rewrite_query": n_rewrite_query, "retrieve": n_retrieve, "grade_documents": n_grade_documents,
    "generate": n_generate, "add_citations": n_add_citations,
}


def build_graph():
    g = StateGraph(State)
    active = [n for n in S["nodes"] if n["enabled"]]
    for n in active:
        g.add_node(n["id"], BUILTIN[n["id"]] if n["kind"] == "builtin" else make_custom(n["instruction"]))
    g.add_edge(START, active[0]["id"])
    for a, b in zip(active, active[1:]):
        g.add_edge(a["id"], b["id"])
    g.add_edge(active[-1]["id"], END)
    return g.compile()


def refresh_graph():
    global _graph
    _graph = build_graph()
    return _graph


def graph_payload():
    g = refresh_graph()
    return {"mermaid": g.get_graph().draw_mermaid(), "nodes": S["nodes"],
            "ready": S["store"] is not None, "files": S["files"]}


# ---------------------------------------------------------------- API
def read_file(name: str, data: bytes) -> str:
    ext = name.lower().rsplit(".", 1)[-1]
    if ext == "pdf":
        from pypdf import PdfReader
        return "\n".join((p.extract_text() or "") for p in PdfReader(io.BytesIO(data)).pages)
    if ext == "docx":
        import docx
        return "\n".join(p.text for p in docx.Document(io.BytesIO(data)).paragraphs)
    return data.decode("utf-8", errors="ignore")


@app.post("/api/upload")
async def upload(files: List[UploadFile] = File(...)):
    docs = []
    for f in files:
        text = read_file(f.filename, await f.read())
        if text.strip():
            docs.append(Document(page_content=text, metadata={"source": f.filename}))
    if not docs:
        raise HTTPException(400, "No readable text in the uploaded files.")
    chunks = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100).split_documents(docs)
    store = InMemoryVectorStore(embeddings())
    store.add_documents(chunks)
    S["store"] = store
    S["files"] = [d.metadata["source"] for d in docs]
    return {**graph_payload(), "chunks": len(chunks)}


@app.get("/api/graph")
def get_graph():
    return graph_payload()


class KeyIn(BaseModel):
    api_key: str


@app.post("/api/key")
def set_key(k: KeyIn):
    S["api_key"] = k.api_key.strip()
    return {"ok": True}


class NodeIn(BaseModel):
    name: str
    instruction: str


@app.post("/api/nodes/custom")
def add_custom(n: NodeIn):
    nid = re.sub(r"\W+", "_", n.name.strip().lower()).strip("_") or "custom"
    if any(x["id"] == nid for x in S["nodes"]):
        nid += "_" + uuid.uuid4().hex[:4]
    S["nodes"].append({"id": nid, "kind": "custom", "core": False, "enabled": True,
                       "instruction": n.instruction, "desc": n.instruction})
    return graph_payload()


@app.post("/api/nodes/{nid}/toggle")
def toggle(nid: str):
    n = next((x for x in S["nodes"] if x["id"] == nid), None)
    if not n:
        raise HTTPException(404, "no such node")
    if n["core"]:
        raise HTTPException(400, "Core nodes (retrieve, generate) can't be removed.")
    n["enabled"] = not n["enabled"]
    return graph_payload()


@app.delete("/api/nodes/{nid}")
def delete(nid: str):
    n = next((x for x in S["nodes"] if x["id"] == nid), None)
    if not n:
        raise HTTPException(404, "no such node")
    if n["core"]:
        raise HTTPException(400, "Core nodes can't be deleted.")
    if n["kind"] == "builtin":
        n["enabled"] = False  # built-ins stay in the palette so they can be re-added
    else:
        S["nodes"].remove(n)
    return graph_payload()


class ChatIn(BaseModel):
    question: str


@app.post("/api/chat")
def chat(c: ChatIn):
    if S["store"] is None:
        raise HTTPException(400, "Upload data first.")
    out = refresh_graph().invoke({"question": c.question})
    return {"answer": out["answer"], "query": out.get("query"),
            "sources": sorted({d.metadata.get("source", "?") for d in out.get("docs", [])})}


@app.get("/")
def index():
    return FileResponse(os.path.join(BASE, "index.html"))
