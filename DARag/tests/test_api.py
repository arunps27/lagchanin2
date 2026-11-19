import pytest
from fastapi.testclient import TestClient

from app.main import create_app


class FakeLLM:
    """Deterministic stand-in for the Groq client."""

    def __init__(self):
        self.route = "sql"

    def complete(self, system, user, **kw):
        if system.startswith("Classify"):
            return f'{{"route": "{self.route}"}}'
        if system.startswith("You are a DuckDB"):
            if "sales_q" in user or "people" in user:
                return "```sql\nSELECT count(*) AS n FROM people\n```"
            return "SELECT department_id, count(*) AS n FROM employees GROUP BY 1 ORDER BY 1"
        return "stub answer"


@pytest.fixture
def client(settings):
    llm = FakeLLM()
    with TestClient(create_app(settings, llm)) as c:
        c.llm = llm
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert "employees" in r.json()["tables"]
    assert r.json()["policy_chunks"] > 0


def test_ask_sql_with_export(client):
    r = client.post("/ask", json={"question": "headcount by department", "export": True})
    body = r.json()
    assert r.status_code == 200 and body["route"] == "sql"
    assert body["answer"] == "stub answer"
    assert body["columns"] == ["department_id", "n"]
    csv_text = client.get(body["export_url"]).text
    assert csv_text.splitlines()[0] == "department_id,n"


def test_ask_docs_cites_policy(client):
    client.llm.route = "docs"
    r = client.post("/ask", json={"question": "How many weeks of parental leave do primary caregivers get?"})
    body = r.json()
    assert body["sql"] is None
    assert body["sources"][0]["source"] == "leave_policy.md"


def test_upload_query_download_roundtrip(client):
    csv_bytes = b"Name,Dept,Score\nAnn,Eng,5\nBob,Ops,3\n=cmd|x,Ops,1\n"
    r = client.post("/datasets/upload", files={"file": ("People 2026.csv", csv_bytes, "text/csv")})
    assert r.status_code == 201
    info = r.json()
    assert info["name"] == "people_2026" and info["rows"] == 3

    # duplicate needs overwrite
    assert client.post("/datasets/upload", files={"file": ("People 2026.csv", csv_bytes)}).status_code == 409
    assert client.post("/datasets/upload?overwrite=true", files={"file": ("People 2026.csv", csv_bytes)}).status_code == 201

    names = {d["name"]: d["kind"] for d in client.get("/datasets").json()}
    assert names["people_2026"] == "upload" and names["employees"] == "builtin"

    dl = client.get("/datasets/people_2026/download")
    assert dl.status_code == 200 and dl.headers["content-type"].startswith("text/csv")
    lines = dl.text.splitlines()
    assert lines[0] == "_name,dept,score"  # normalised headers (DuckDB prefixes reserved word 'name')
    assert "'=cmd|x" in dl.text  # formula injection neutralised

    assert client.delete("/datasets/people_2026").status_code == 200
    assert client.get("/datasets/people_2026/download").status_code == 404


def test_upload_validation(client):
    assert client.post("/datasets/upload", files={"file": ("x.txt", b"a,b\n1,2\n")}).status_code == 400
    assert client.post("/datasets/upload", files={"file": ("x.csv", b"  \n")}).status_code == 400
    assert client.post("/datasets/upload", files={"file": ("x.csv", b"a,b\n")}).status_code == 400
    assert client.post("/datasets/upload", files={"file": ("employees.csv", b"a\n1\n")}).status_code == 409
    assert client.delete("/datasets/employees").status_code == 403


def test_builtin_download_masks_sensitive(client):
    text = client.get("/datasets/employees/download").text
    header, first = text.splitlines()[:2]
    cols = header.split(",")
    assert "***" in first and cols.index("base_salary") >= 0
