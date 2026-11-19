import json
import re
from dataclasses import dataclass
from datetime import date

import duckdb

from . import pii
from .config import Settings
from .db import Database

BLOCKED_FUNC_PREFIXES = ("read_", "scan_", "parquet_", "glob", "getenv", "current_setting", "query", "sniff_", "load_")

METRIC_HINTS = """Definitions:
- Headcount = employees with employment_status = 'Active' (as of today). Historical headcount at date D = hire_date <= D AND (termination_date IS NULL OR termination_date > D).
- Attrition (terminations) in a period = employees whose termination_date falls in that period.
- Attrition rate = terminations in period / average headcount in period (use headcount at period start and end).
- Tenure = years between hire_date and COALESCE(termination_date, today).
- Time to fill = filled_date - opened_date (days) in job_requisitions."""


class SQLValidationError(ValueError):
    pass


@dataclass
class QueryResult:
    sql: str
    columns: list[str]
    rows: list[tuple]
    truncated: bool


def _walk(node, visit):
    if isinstance(node, dict):
        visit(node)
        for v in node.values():
            _walk(v, visit)
    elif isinstance(node, list):
        for v in node:
            _walk(v, visit)


def parse_and_validate(db: Database, sql: str) -> tuple[str, dict]:
    """Validate that sql is a single read-only SELECT over allowed tables. Returns (clean_sql, ast)."""
    cleaned = sql.strip().rstrip(";").strip()
    if not cleaned:
        raise SQLValidationError("Empty SQL.")
    try:
        with db.lock:
            raw = db.con.execute("SELECT json_serialize_sql(?)", [cleaned]).fetchone()[0]
    except duckdb.Error as exc:
        raise SQLValidationError(f"Could not parse SQL: {exc}") from exc
    ast = json.loads(raw)
    if ast.get("error"):
        raise SQLValidationError(f"Only a single read-only SELECT statement is allowed ({ast.get('error_message', '')}).")
    if len(ast.get("statements", [])) != 1:
        raise SQLValidationError("Exactly one SELECT statement is allowed.")

    allowed = {t.lower() for t in db.tables()}
    ctes: set[str] = set()
    problems: list[str] = []

    def collect_ctes(n):
        if "cte_map" in n:
            for e in (n["cte_map"] or {}).get("map", []):
                ctes.add(str(e["key"]).lower())

    _walk(ast, collect_ctes)

    def check(n):
        t = n.get("type")
        if t == "TABLE_FUNCTION":
            problems.append("table functions are not allowed")
        elif t == "BASE_TABLE":
            name = str(n.get("table_name", "")).lower()
            if n.get("catalog_name") or n.get("schema_name") not in ("", "main", None):
                problems.append(f"schema-qualified table '{name}' is not allowed")
            elif name not in allowed and name not in ctes:
                problems.append(f"unknown table '{name}'")
        elif t == "FUNCTION":
            fn = str(n.get("function_name", "")).lower()
            if fn.startswith(BLOCKED_FUNC_PREFIXES):
                problems.append(f"function '{fn}' is not allowed")

    _walk(ast, check)
    if problems:
        raise SQLValidationError("; ".join(sorted(set(problems))) + f". Allowed tables: {', '.join(sorted(allowed))}")
    return cleaned, ast


def run_query(db: Database, sql: str, settings: Settings) -> QueryResult:
    cleaned, ast = parse_and_validate(db, sql)
    positions, star = set(), False
    if not settings.allow_sensitive:
        try:
            positions, star = pii.analyze(ast)
        except pii.SensitiveColumnError as exc:
            raise SQLValidationError(str(exc)) from exc
    try:
        cols, rows, truncated = db.execute(cleaned, max_rows=settings.max_result_rows)
    except duckdb.Error as exc:
        raise SQLValidationError(f"Query failed: {exc}") from exc
    if not settings.allow_sensitive:
        rows = pii.mask_rows(cols, rows, positions, star)
    return QueryResult(cleaned, cols, rows, truncated)


def extract_sql(text: str) -> str:
    m = re.search(r"```(?:sql)?\s*(.*?)```", text, re.S | re.I)
    return (m.group(1) if m else text).strip()


SQL_SYSTEM = """You are a DuckDB SQL expert for HR analytics.
Write ONE read-only SELECT statement (CTEs allowed) that answers the question.
Rules:
- Output only the SQL, no explanation, no markdown.
- Answer only the data part of the question. Never add literal text columns about policies, advice or explanations; policy questions are handled separately.
- Use only the tables and columns listed in the schema; quote nothing unless needed.
- Never use table functions or file access. Do not select raw salary/email/phone columns; use aggregates for pay.
- Prefer clear column aliases. Use ROUND for rates/averages. Add ORDER BY for rankings.
- Today's date is {today}; interpret 'last year', 'this year', 'last quarter' relative to it.

{hints}

Schema:
{schema}"""


def generate_sql(llm, db: Database, question: str, previous_sql: str | None = None, error: str | None = None) -> str:
    system = SQL_SYSTEM.format(today=date.today().isoformat(), hints=METRIC_HINTS, schema=db.schema_text())
    user = f"Question: {question}"
    if previous_sql and error:
        user += f"\n\nYour previous SQL failed.\nSQL: {previous_sql}\nError: {error}\nReturn a corrected SELECT statement."
    return extract_sql(llm.complete(system, user, reasoning_effort="medium"))
