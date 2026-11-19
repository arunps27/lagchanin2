import json, os, sqlite3
from pathlib import Path
from fastapi import FastAPI, Body
from fastapi.responses import StreamingResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from .workflow import build_app, ROOT, DB, MODEL
from . import seed

if not DB.exists():
    seed.build()
graph_app = build_app()
api = FastAPI(title="Loan KYC Graph RAG")


@api.get("/api/applications")
def apps():
    c = sqlite3.connect(DB); c.row_factory = sqlite3.Row
    rows = [dict(r) for r in c.execute(
        "SELECT a.app_id,a.amount,a.purpose,c.name FROM applications a JOIN customers c ON c.id=a.customer_id")]
    return {"applications": rows, "model": MODEL, "groq_key_set": bool(os.environ.get("GROQ_API_KEY"))}


@api.post("/api/key")
def set_key(body: dict = Body(...)):
    os.environ["GROQ_API_KEY"] = body.get("key", "").strip()
    return {"ok": True}


@api.get("/api/run/{app_id}")
async def run(app_id: str):
    async def gen():
        try:
            if not os.environ.get("GROQ_API_KEY"):
                raise RuntimeError("GROQ_API_KEY is not set (put it in .env or paste it in the UI)")
            async for mode, chunk in graph_app.astream({"app_id": app_id}, stream_mode=["custom", "updates"]):
                if mode == "custom":
                    yield f"data: {json.dumps(chunk, default=str)}\n\n"
                else:
                    for node, upd in chunk.items():
                        keys = [k for k in (upd or {}).keys() if k not in ("graph", "rows", "doc_text")]
                        yield f"data: {json.dumps({'type': 'node_done', 'node': node, 'keys': keys})}\n\n"
                        if node in ("underwrite", "manual_review", "compliance_reject"):
                            yield f"data: {json.dumps({'type': 'final', **upd['decision']}, default=str)}\n\n"
        except Exception as e:  # surface errors live in the UI
            yield f"data: {json.dumps({'type': 'error', 'message': f'{type(e).__name__}: {e}'})}\n\n"
        yield f"data: {json.dumps({'type': 'done'})}\n\n"
    return StreamingResponse(gen(), media_type="text/event-stream")


@api.get("/")
def index():
    return FileResponse(Path(__file__).parent / "static" / "index.html")
