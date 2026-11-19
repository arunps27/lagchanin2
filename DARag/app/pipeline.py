import json
import re

from . import sql_engine
from .config import Settings
from .db import Database
from .doc_store import DocStore
from .sql_engine import SQLValidationError

ROUTER_SYSTEM = """Classify an HR question into how it should be answered.
- "sql": needs numbers/aggregations/lists from HR data tables (headcount, attrition, payroll, leave, hiring, uploaded datasets).
- "docs": asks about HR policy, rules, entitlements or procedures found in policy documents.
- "both": needs data AND policy/context.
Reply with JSON only: {"route": "sql" | "docs" | "both"}"""

ANSWER_SYSTEM = """You are an HR operations analytics assistant.
Answer using ONLY the data results and policy excerpts provided. Be concise and precise with numbers.
- Cite policy facts as [source: filename]. Policy statements may come ONLY from the policy excerpts, never from SQL result columns or your own knowledge.
- If data or policy is missing or a query failed, say so plainly; never invent figures.
- State assumptions (time period, definitions) briefly.
- Never reveal masked or personal values; report aggregates."""


def classify(llm, question: str) -> str:
    try:
        text = llm.complete(ROUTER_SYSTEM, question, reasoning_effort="low", max_tokens=512, temperature=0)
        m = re.search(r'"route"\s*:\s*"(sql|docs|both)"', text) or re.search(r"\b(sql|docs|both)\b", text.lower())
        return m.group(1) if m else "both"
    except Exception:
        return "both"


def _run_sql_path(llm, db: Database, settings: Settings, question: str) -> dict:
    sql, error, result = None, None, None
    for attempt in range(2):
        try:
            sql = sql_engine.generate_sql(llm, db, question, previous_sql=sql if attempt else None, error=error)
            result = sql_engine.run_query(db, sql, settings)
            error = None
            break
        except SQLValidationError as exc:
            error = str(exc)
    return {"sql": sql, "result": result, "error": error}


def ask(llm, db: Database, docs: DocStore, settings: Settings, question: str) -> dict:
    route = classify(llm, question)
    sql_out = None
    sources: list[dict] = []
    parts = [f"Question: {question}"]

    if route in ("sql", "both"):
        sql_out = _run_sql_path(llm, db, settings, question)
        if sql_out["result"]:
            r = sql_out["result"]
            shown = r.rows[: settings.rows_to_llm]
            parts.append(
                f"SQL used:\n{r.sql}\n\nResult ({len(r.rows)} rows{', truncated' if r.truncated else ''}; "
                f"showing {len(shown)}):\ncolumns={r.columns}\nrows={json.dumps(shown, default=str)}"
            )
        else:
            parts.append(f"The data query failed: {sql_out['error']}")

    if route in ("docs", "both"):
        sources = docs.search(question, settings.retrieval_k)
        if sources:
            parts.append("Policy excerpts:\n" + "\n---\n".join(s["text"] for s in sources))
        else:
            parts.append("No policy documents matched.")

    answer = llm.complete(ANSWER_SYSTEM, "\n\n".join(parts), reasoning_effort="medium")
    result = sql_out["result"] if sql_out else None
    return {
        "answer": answer,
        "route": route,
        "sql": sql_out["sql"] if sql_out else None,
        "sql_error": sql_out["error"] if sql_out else None,
        "columns": result.columns if result else None,
        "rows": [list(r) for r in result.rows[: settings.rows_to_llm]] if result else None,
        "row_count": len(result.rows) if result else None,
        "truncated": result.truncated if result else None,
        "sources": [{"source": s["source"], "section": s["section"], "score": s["score"]} for s in sources],
        "_result": result,
    }
