import threading
from pathlib import Path

import duckdb

BUILTIN_TABLES = {"departments", "employees", "leave_requests", "payroll", "job_requisitions"}

TABLE_DOCS = {
    "departments": "One row per department.",
    "employees": "One row per employee ever employed. Terminated employees have termination_date set.",
    "leave_requests": "Employee leave requests.",
    "payroll": "Monthly payroll record per active employee.",
    "job_requisitions": "Open and filled hiring requisitions.",
}

COLUMN_DOCS = {
    "employees.employment_status": "'Active' or 'Terminated'",
    "employees.termination_reason": "e.g. Resignation, Layoff, Retirement, Dismissal",
    "employees.performance_rating": "1 (low) to 5 (high), latest review",
    "employees.base_salary": "annual base salary in USD (sensitive)",
    "employees.manager_id": "employee_id of the manager",
    "leave_requests.days": "working days requested",
    "payroll.pay_month": "first day of the pay month",
    "job_requisitions.filled_date": "NULL while open",
}


class Database:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.con = duckdb.connect(str(self.path))
        self.lock = threading.RLock()
        self.con.execute(
            "CREATE TABLE IF NOT EXISTS _catalog("
            "name VARCHAR PRIMARY KEY, kind VARCHAR, filename VARCHAR, uploaded_at TIMESTAMP)"
        )

    def execute(self, sql: str, params=None, max_rows: int | None = None):
        """Run a statement; returns (columns, rows, truncated)."""
        with self.lock:
            cur = self.con.cursor()
            try:
                cur.execute(sql, params or [])
                cols = [d[0] for d in cur.description] if cur.description else []
                if not cols:
                    return cols, [], False
                if max_rows is None:
                    return cols, cur.fetchall(), False
                rows = cur.fetchmany(max_rows + 1)
                truncated = len(rows) > max_rows
                return cols, rows[:max_rows], truncated
            finally:
                cur.close()

    def tables(self) -> list[str]:
        _, rows, _ = self.execute(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema='main' AND NOT starts_with(table_name,'_') ORDER BY table_name"
        )
        return [r[0] for r in rows]

    def columns(self, table: str) -> list[tuple[str, str]]:
        _, rows, _ = self.execute(
            "SELECT column_name, data_type FROM information_schema.columns "
            "WHERE table_schema='main' AND table_name=? ORDER BY ordinal_position",
            [table],
        )
        return [(r[0], r[1]) for r in rows]

    def row_count(self, table: str) -> int:
        _, rows, _ = self.execute(f'SELECT count(*) FROM "{table}"')
        return rows[0][0]

    def schema_text(self) -> str:
        lines = []
        for t in self.tables():
            lines.append(f'TABLE "{t}" ({self.row_count(t)} rows)' + (f" -- {TABLE_DOCS[t]}" if t in TABLE_DOCS else ""))
            for name, dtype in self.columns(t):
                note = COLUMN_DOCS.get(f"{t}.{name}", "")
                if dtype == "VARCHAR":
                    _, vals, _ = self.execute(f'SELECT DISTINCT "{name}" FROM "{t}" WHERE "{name}" IS NOT NULL LIMIT 13')
                    if 0 < len(vals) <= 12:
                        note = (note + " " if note else "") + "values: " + ", ".join(repr(v[0]) for v in vals)
                lines.append(f'  "{name}" {dtype}' + (f" -- {note}" if note else ""))
        return "\n".join(lines)

    def has_table(self, name: str) -> bool:
        return name.lower() in {t.lower() for t in self.tables()}
