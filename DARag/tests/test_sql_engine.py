import pytest

from app import synthetic
from app.sql_engine import SQLValidationError, run_query


@pytest.mark.parametrize(
    "sql",
    [
        "DROP TABLE employees",
        "UPDATE employees SET base_salary = 0",
        "DELETE FROM employees",
        "SELECT 1; SELECT 2",
        "SELECT * FROM read_csv('x.csv')",
        "SELECT * FROM 'x.csv'",
        "SELECT * FROM _catalog",
        "SELECT * FROM unknown_table",
        "SELECT * FROM information_schema.tables",
        "COPY employees TO 'x.csv'",
        "ATTACH 'other.db'",
        "",
    ],
)
def test_rejects_unsafe_sql(db, settings, sql):
    with pytest.raises(SQLValidationError):
        run_query(db, sql, settings)


def test_allows_select_and_cte(db, settings):
    r = run_query(
        db,
        "WITH t AS (SELECT department_id, count(*) n FROM employees GROUP BY 1) SELECT * FROM t ORDER BY n DESC;",
        settings,
    )
    assert r.columns == ["department_id", "n"]
    assert sum(row[1] for row in r.rows) == 80


def test_sensitive_columns_masked(db, settings):
    r = run_query(db, "SELECT employee_id, base_salary, email FROM employees LIMIT 3", settings)
    assert all(row[1] == "***" and row[2] == "***" for row in r.rows)
    r = run_query(db, "SELECT * FROM employees LIMIT 3", settings)
    i = r.columns.index("base_salary")
    assert all(row[i] == "***" for row in r.rows)
    r = run_query(db, "SELECT base_salary + 0 AS s FROM employees LIMIT 2", settings)
    assert all(row[0] == "***" for row in r.rows)


def test_aggregates_of_sensitive_columns_allowed(db, settings):
    r = run_query(db, "SELECT department_id, round(avg(base_salary)) AS avg_pay FROM employees GROUP BY 1", settings)
    assert all(isinstance(row[1], float) for row in r.rows)


def test_nested_sensitive_select_rejected(db, settings):
    with pytest.raises(SQLValidationError):
        run_query(db, "SELECT s FROM (SELECT base_salary AS s FROM employees)", settings)


def test_synthetic_is_deterministic():
    a = synthetic.generate_tables(50)
    b = synthetic.generate_tables(50)
    assert a["employees"][1] == b["employees"][1]
    assert len(a["employees"][1]) == 50
