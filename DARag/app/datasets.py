import csv
import io
import re
import secrets
import threading
import time
import uuid
from pathlib import Path

import duckdb

from . import pii
from .config import Settings
from .db import BUILTIN_TABLES, Database


class DatasetError(ValueError):
    def __init__(self, message: str, status: int = 400):
        super().__init__(message)
        self.status = status


def sanitize_table_name(filename: str) -> str:
    stem = Path(filename).stem.lower()
    name = re.sub(r"[^a-z0-9]+", "_", stem).strip("_")[:50]
    if not name:
        name = "dataset"
    if name[0].isdigit():
        name = "t_" + name
    return name


def upload_csv(db: Database, settings: Settings, filename: str, data: bytes, overwrite: bool = False) -> dict:
    if not filename.lower().endswith(".csv"):
        raise DatasetError("Only .csv files are accepted.")
    if not data.strip():
        raise DatasetError("The file is empty.")
    if len(data) > settings.max_upload_mb * 1024 * 1024:
        raise DatasetError(f"File exceeds {settings.max_upload_mb} MB.", 413)

    name = sanitize_table_name(filename)
    if name in BUILTIN_TABLES:
        raise DatasetError(f"'{name}' is a built-in HR table; rename the file.", 409)
    if db.has_table(name) and not overwrite:
        raise DatasetError(f"Dataset '{name}' already exists (pass overwrite=true to replace).", 409)

    settings.uploads_dir.mkdir(parents=True, exist_ok=True)
    path = settings.uploads_dir / f"{uuid.uuid4().hex}.csv"
    path.write_bytes(data)
    try:
        with db.lock:
            db.con.execute(
                f'CREATE OR REPLACE TABLE "{name}" AS '
                "SELECT * FROM read_csv(?, header=true, auto_detect=true, normalize_names=true, sample_size=-1)",
                [str(path)],
            )
            cols = db.columns(name)
            rows = db.row_count(name)
            if not cols or rows == 0:
                db.con.execute(f'DROP TABLE IF EXISTS "{name}"')
                raise DatasetError("CSV has no data rows.")
            db.con.execute("DELETE FROM _catalog WHERE name=?", [name])
            db.con.execute("INSERT INTO _catalog VALUES (?, 'upload', ?, now())", [name, Path(filename).name])
    except duckdb.Error as exc:
        raise DatasetError(f"Could not parse CSV: {exc}") from exc
    finally:
        path.unlink(missing_ok=True)

    sensitive = [c for c, _ in cols if c.lower() in pii.SENSITIVE_COLUMNS]
    return {"name": name, "rows": rows, "columns": [{"name": c, "type": t} for c, t in cols], "sensitive_columns": sensitive}


def list_datasets(db: Database) -> list[dict]:
    _, cat, _ = db.execute("SELECT name, filename, uploaded_at FROM _catalog")
    uploaded = {r[0]: r for r in cat}
    out = []
    for t in db.tables():
        out.append(
            {
                "name": t,
                "kind": "builtin" if t in BUILTIN_TABLES else "upload",
                "rows": db.row_count(t),
                "columns": [{"name": c, "type": ty} for c, ty in db.columns(t)],
                "filename": uploaded[t][1] if t in uploaded else None,
            }
        )
    return out


def delete_dataset(db: Database, name: str) -> None:
    if name in BUILTIN_TABLES:
        raise DatasetError("Built-in HR tables cannot be deleted.", 403)
    if not db.has_table(name):
        raise DatasetError(f"Dataset '{name}' not found.", 404)
    with db.lock:
        db.con.execute(f'DROP TABLE "{name}"')
        db.con.execute("DELETE FROM _catalog WHERE name=?", [name])


def csv_safe(value):
    """Neutralise spreadsheet formula injection."""
    if isinstance(value, str) and value[:1] in ("=", "+", "-", "@", "\t", "\r"):
        return "'" + value
    return value


def iter_csv(columns: list[str], rows):
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow([csv_safe(c) for c in columns])
    yield buf.getvalue()
    for r in rows:
        buf.seek(0)
        buf.truncate()
        w.writerow([csv_safe(v) for v in r])
        yield buf.getvalue()


def table_for_download(db: Database, settings: Settings, name: str):
    if not db.has_table(name):
        raise DatasetError(f"Dataset '{name}' not found.", 404)
    real = next(t for t in db.tables() if t.lower() == name.lower())
    cols, rows, _ = db.execute(f'SELECT * FROM "{real}"')
    # Built-in HR tables honour the sensitive-column policy; the user's own uploads are returned as-is.
    if real in BUILTIN_TABLES and not settings.allow_sensitive:
        rows = pii.mask_rows(cols, rows, set(), True)
    return real, cols, rows


class ExportStore:
    """Short-lived in-memory store for query results that can be downloaded as CSV."""

    def __init__(self, ttl: int = 900, max_items: int = 50):
        self.ttl, self.max_items = ttl, max_items
        self._items: dict[str, tuple[float, list[str], list[tuple]]] = {}
        self._lock = threading.Lock()

    def put(self, columns, rows) -> str:
        with self._lock:
            now = time.time()
            self._items = {k: v for k, v in self._items.items() if now - v[0] < self.ttl}
            while len(self._items) >= self.max_items:
                self._items.pop(next(iter(self._items)))
            key = secrets.token_urlsafe(12)
            self._items[key] = (now, list(columns), list(rows))
            return key

    def get(self, key: str):
        with self._lock:
            item = self._items.get(key)
            if not item or time.time() - item[0] >= self.ttl:
                return None
            return item[1], item[2]
