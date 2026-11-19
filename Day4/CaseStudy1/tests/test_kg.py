import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from kg import KnowledgeGraph, max_severity, name_similarity  # noqa: E402
from nodes import credit_metrics  # noqa: E402

kg = KnowledgeGraph()


def test_clean_applicant_has_no_findings():
    assert kg.all_findings("P-001", "Priya Sharma", "1988-04-02") == []


def test_fraud_ring_shared_phone_is_high():
    f = kg.shared_identifier_findings("P-004")
    assert max_severity(f) == "HIGH"
    assert any("P-005" in e for x in f for e in x["evidence"])


def test_sanctions_near_match():
    f = kg.watchlist_findings("Ivan Petrovich Sokolov", "1975-03-12")
    assert f and f[0]["severity"] == "HIGH" and "MATCHES" in f[0]["description"]
    assert name_similarity("Sokolov Ivan Petrovic", "Ivan Petrovic Sokolov") == 1.0


def test_pep_relative_is_medium():
    f = kg.all_findings("P-008", "Aisha Rahman", "1989-08-08")
    assert max_severity(f) == "MEDIUM"
    assert any(x["type"] == "RELATED_TO_PEP" for x in f)


def _app(**kw):
    base = dict(monthly_income=9000, existing_monthly_debt=800, loan_amount=150000, loan_term_months=240,
                credit_score=760, credit_history_years=12, property_value=220000)
    return {**base, **kw}


def test_credit_rules():
    assert credit_metrics(_app())["rule_outcome"] == "APPROVE"
    assert credit_metrics(_app(credit_history_years=1))["rule_outcome"] == "REFER"
    assert credit_metrics(_app(monthly_income=4000, existing_monthly_debt=1800))["rule_outcome"] == "DECLINE"
