"""Column-level guard for sensitive HR fields.

Raw values of sensitive columns are masked in query output unless they appear only inside
aggregates (AVG(salary), SUM(bonus), ...). This is a guardrail, not a security boundary.
"""

SENSITIVE_COLUMNS = {
    "salary", "base_salary", "bonus", "gross_pay", "net_pay", "deductions",
    "email", "phone", "ssn", "dob", "date_of_birth", "address", "national_id",
}

AGGREGATES = {
    "avg", "mean", "sum", "count", "min", "max", "median", "stddev", "stddev_pop", "stddev_samp",
    "variance", "var_pop", "var_samp", "quantile", "quantile_cont", "quantile_disc", "mode",
    "approx_quantile", "percentile_cont", "percentile_disc", "bool_and", "bool_or", "sum_no_overflow",
}

MASK = "***"


class SensitiveColumnError(ValueError):
    pass


def _exposes(node) -> bool:
    """True if the expression exposes a sensitive column outside an aggregate."""
    if isinstance(node, dict):
        t = node.get("type")
        if t in ("FUNCTION", "WINDOW") and str(node.get("function_name", "")).lower() in AGGREGATES:
            return False
        if t == "COLUMN_REF":
            names = node.get("column_names") or []
            return bool(names) and str(names[-1]).lower() in SENSITIVE_COLUMNS
        return any(_exposes(v) for v in node.values())
    if isinstance(node, list):
        return any(_exposes(v) for v in node)
    return False


def _top_selects(node) -> list[dict]:
    if not isinstance(node, dict):
        return []
    if node.get("type") == "SELECT_NODE":
        return [node]
    out = []
    for key in ("left", "right", "children"):
        v = node.get(key)
        for item in v if isinstance(v, list) else [v]:
            out += _top_selects(item)
    return out


def _all_select_nodes(node):
    if isinstance(node, dict):
        if "select_list" in node:
            yield node
        for v in node.values():
            yield from _all_select_nodes(v)
    elif isinstance(node, list):
        for v in node:
            yield from _all_select_nodes(v)


def analyze(ast: dict) -> tuple[set[int], bool]:
    """Return (positions to mask, star_present). Raises if nested selects expose sensitive columns."""
    root = ast["statements"][0]["node"]
    top = _top_selects(root)
    top_ids = {id(n) for n in top}
    positions: set[int] = set()
    star = False
    for sel in _all_select_nodes(root):
        is_top = id(sel) in top_ids
        for i, item in enumerate(sel["select_list"]):
            if item.get("type") == "STAR":
                star = star or is_top
            elif _exposes(item):
                if is_top:
                    positions.add(i)
                else:
                    raise SensitiveColumnError(
                        "Raw sensitive columns (salary, email, ...) cannot be selected inside subqueries; "
                        "use aggregates such as AVG(base_salary)."
                    )
    return positions, star


def mask_rows(columns: list[str], rows: list[tuple], positions: set[int], star: bool) -> list[tuple]:
    idx = set(positions)
    if star:
        idx |= {i for i, c in enumerate(columns) if c.lower() in SENSITIVE_COLUMNS}
    if not idx:
        return [tuple(r) for r in rows]
    return [tuple(MASK if i in idx and v is not None else v for i, v in enumerate(r)) for r in rows]
