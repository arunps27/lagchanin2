# HR Analytics RAG

FastAPI service that answers HR operations questions by routing each one to:

- **SQL path** – text-to-SQL over DuckDB (headcount, attrition, payroll, leave, hiring, plus any uploaded CSV)
- **Docs path** – vector search over HR policy documents (Chroma, local embeddings)
- **Both** – data plus policy, synthesised into one cited answer

LLM: Groq `openai/gpt-oss-120b`. Synthetic HR data and sample policies are generated on first start.

## Run

```powershell
python -m venv .venv
.\.venv\Scripts\pip install -r requirements.txt
copy .env.example .env      # then put your GROQ_API_KEY in .env
.\.venv\Scripts\uvicorn app.main:app --reload
```

Interactive docs: http://127.0.0.1:8000/docs

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| POST | `/ask` `{question, export?}` | Answer; returns route, SQL, rows, sources; `export:true` adds a CSV link |
| GET | `/exports/{id}` | Download a query result as CSV (expires after 15 min) |
| POST | `/datasets/upload` | Upload a CSV (multipart `file`, `?overwrite=true`) – becomes a queryable table |
| GET | `/datasets` | List tables and schemas |
| GET | `/datasets/{name}/download` | Download a table as CSV |
| DELETE | `/datasets/{name}` | Delete an uploaded table |
| POST | `/ingest` | Re-index the policy folder |
| POST | `/documents/upload` | Add a `.md`/`.txt` policy document |
| GET | `/schema`, `/health` | Introspection |

```powershell
curl -X POST localhost:8000/ask -H "Content-Type: application/json" `
  -d '{"question":"What was attrition by department in 2025?","export":true}'
curl -F "file=@candidates.csv" localhost:8000/datasets/upload
```

## Guardrails

- Generated SQL must parse as a single SELECT over known tables (AST check via DuckDB); table functions, file access, DDL/DML are rejected. One automatic repair retry on failure. Result rows capped at 5000.
- Sensitive columns (salary, email, phone, ...) are masked in row-level output; aggregates such as `AVG(base_salary)` work. Set `ALLOW_SENSITIVE=true` to disable. This is a guardrail, not a security boundary; add authentication before exposing the service.
- CSV downloads neutralise spreadsheet formula injection (`=`, `+`, `-`, `@`). Built-in HR table downloads honour the masking policy; your own uploads are returned unchanged.

## Tests

```powershell
.\.venv\Scripts\pytest
```
Tests use a fake LLM; no API key needed.
