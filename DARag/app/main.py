from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from . import datasets, pipeline, synthetic
from .config import Settings, get_settings
from .db import BUILTIN_TABLES, Database
from .doc_store import DocStore
from .llm import LLM, LLMError


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000)
    export: bool = False


def create_app(settings: Settings | None = None, llm=None) -> FastAPI:
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        db = Database(settings.db_path)
        if not BUILTIN_TABLES.issubset({t.lower() for t in db.tables()}):
            synthetic.seed_database(db)
        if not settings.policies_dir.exists() or not any(settings.policies_dir.glob("*.md")):
            synthetic.write_policies(settings.policies_dir)
        docs = DocStore(settings.chroma_dir, settings.policies_dir)
        if docs.count() == 0:
            docs.reindex()
        app.state.db, app.state.docs = db, docs
        app.state.llm = llm or LLM(settings.groq_api_key, settings.llm_model)
        app.state.exports = datasets.ExportStore(settings.export_ttl_seconds)
        yield

    app = FastAPI(title="HR Analytics RAG", version="0.1.0", lifespan=lifespan)

    @app.exception_handler(datasets.DatasetError)
    async def dataset_error(_: Request, exc: datasets.DatasetError):
        return JSONResponse({"detail": str(exc)}, status_code=exc.status)

    @app.exception_handler(LLMError)
    async def llm_error(_: Request, exc: LLMError):
        return JSONResponse({"detail": str(exc)}, status_code=502)

    @app.get("/", include_in_schema=False)
    def ui():
        return FileResponse(Path(__file__).parent / "static" / "index.html")

    @app.get("/health")
    def health(request: Request):
        s = request.app.state
        return {"status": "ok", "tables": s.db.tables(), "policy_chunks": s.docs.count(), "model": settings.llm_model}

    @app.get("/schema")
    def schema(request: Request):
        return {"schema": request.app.state.db.schema_text()}

    @app.post("/ask")
    def ask(body: AskRequest, request: Request):
        s = request.app.state
        out = pipeline.ask(s.llm, s.db, s.docs, settings, body.question)
        result = out.pop("_result")
        if body.export and result:
            export_id = s.exports.put(result.columns, result.rows)
            out["export_id"] = export_id
            out["export_url"] = f"/exports/{export_id}"
        return out

    @app.get("/exports/{export_id}")
    def download_export(export_id: str, request: Request):
        item = request.app.state.exports.get(export_id)
        if not item:
            raise HTTPException(404, "Export not found or expired.")
        cols, rows = item
        return StreamingResponse(
            datasets.iter_csv(cols, rows),
            media_type="text/csv",
            headers={"Content-Disposition": 'attachment; filename="query_result.csv"'},
        )

    @app.post("/datasets/upload", status_code=201)
    def upload_dataset(request: Request, file: UploadFile = File(...), overwrite: bool = Query(False)):
        limit = settings.max_upload_mb * 1024 * 1024
        data = file.file.read(limit + 1)
        return datasets.upload_csv(request.app.state.db, settings, file.filename or "", data, overwrite)

    @app.get("/datasets")
    def list_datasets(request: Request):
        return datasets.list_datasets(request.app.state.db)

    @app.delete("/datasets/{name}")
    def delete_dataset(name: str, request: Request):
        datasets.delete_dataset(request.app.state.db, name)
        return {"deleted": name}

    @app.get("/datasets/{name}/download")
    def download_dataset(name: str, request: Request):
        real, cols, rows = datasets.table_for_download(request.app.state.db, settings, name)
        return StreamingResponse(
            datasets.iter_csv(cols, rows),
            media_type="text/csv",
            headers={"Content-Disposition": f'attachment; filename="{real}.csv"'},
        )

    @app.post("/ingest")
    def ingest(request: Request):
        return request.app.state.docs.reindex()

    @app.post("/documents/upload", status_code=201)
    def upload_document(request: Request, file: UploadFile = File(...)):
        name = (file.filename or "").replace("\\", "/").split("/")[-1]
        if not name.lower().endswith((".md", ".txt")):
            raise HTTPException(400, "Only .md or .txt policy documents are accepted.")
        data = file.file.read(2 * 1024 * 1024 + 1)
        if len(data) > 2 * 1024 * 1024:
            raise HTTPException(413, "Document exceeds 2 MB.")
        text = data.decode("utf-8", errors="replace")
        settings.policies_dir.mkdir(parents=True, exist_ok=True)
        (settings.policies_dir / name).write_text(text, encoding="utf-8")
        return {"document": name, "chunks": request.app.state.docs.add_document(name, text)}

    return app


app = create_app()
