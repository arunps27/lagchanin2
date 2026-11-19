"""Generate synthetic policy PDFs (20/40/60/80 pages) and two application files.

  python generate_samples.py      ->  sample_uploads/
Every page holds exactly one section, so page counts are exact.
"""
import csv
import itertools
import json
import random
from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer

OUT = Path(__file__).parent / "sample_uploads"
OUT.mkdir(exist_ok=True)
SS = getSampleStyleSheet()
H1 = ParagraphStyle("h1", parent=SS["Heading1"], fontSize=17, spaceAfter=10)
H2 = ParagraphStyle("h2", parent=SS["Heading2"], fontSize=12, textColor="#444444")
BODY = ParagraphStyle("b", parent=SS["BodyText"], fontSize=10.5, leading=15, spaceAfter=8)

# ---- content banks -------------------------------------------------------
DOCS = {
    20: ("Consumer Lending Credit Policy Manual", "LP", [
        "Credit Eligibility", "Debt-to-Income Assessment", "Credit Score Bands", "Thin-File Applicants",
        "Collateral and Loan-to-Value", "Income Verification", "Employment Stability", "Self-Employed Applicants",
        "Co-signers and Guarantors", "Loan Pricing and Risk Tiers", "Exceptions and Overrides",
        "Manual Underwriting Referral", "Adverse Action Notices", "Fair Lending", "Portfolio Concentration Limits",
        "Early Delinquency Signals", "Model Governance", "Documentation Standards", "Record Retention",
        "Policy Review and Version Control"]),
    40: ("KYC and Customer Due Diligence Procedures", "KP", [
        "Customer Identification Program", "Acceptable Identity Documents", "Address Verification",
        "Date of Birth and Name Matching", "Customer Risk Rating Methodology", "LOW Risk Customers",
        "MEDIUM Risk Customers", "HIGH Risk Customers", "Enhanced Due Diligence Triggers",
        "Source of Funds", "Source of Wealth", "Politically Exposed Persons", "PEP Family and Close Associates",
        "Shared Phone Numbers", "Shared Addresses", "Shared Bank Accounts and Devices", "Synthetic Identity Indicators",
        "Thin Identity Footprint", "Recent Address Changes", "Remote Onboarding Controls", "Document Authenticity Checks",
        "Biometric Verification", "Beneficial Ownership", "Business Applicants", "Third-Party Introducers",
        "Ongoing Monitoring", "Periodic Review Cycles", "Trigger-Event Reviews", "Customer Exit Procedures",
        "Data Quality Controls", "Graph-Based Relationship Analysis", "Hop-Distance Escalation Rules",
        "Analyst Case Management", "Quality Assurance Sampling", "Training and Competency", "Third-Party Data Vendors",
        "Privacy and Data Protection", "Cross-Border Customers", "Vulnerable Customers", "Procedure Governance"]),
    60: ("AML, Sanctions and Fraud Operations Handbook", "AO", [
        "AML Program Overview", "Three Lines of Defence", "Sanctions Screening Scope", "Name Matching Algorithms",
        "Fuzzy Match Thresholds", "Date of Birth Corroboration", "False Positive Management", "Freeze and Hold Actions",
        "No Tipping-Off Rules", "Escalation to Compliance", "Two-Day Review SLA", "Analyst Decision Standards",
        "Suspicious Activity Reporting", "SAR Narrative Standards", "SAR Confidentiality", "Law Enforcement Requests",
        "First-Party Fraud", "Third-Party Fraud", "Fraud Rings and Mule Networks", "Shared Identifier Analytics",
        "Fraud Flag Propagation", "Confirmed Fraud Records", "Application Velocity Checks", "Device Intelligence",
        "Email and Phone Risk Signals", "Address Risk Signals", "Employer Verification", "Income Document Fraud",
        "Identity Theft Response", "Account Takeover", "Transaction Monitoring Rules", "Alert Tuning",
        "Model Validation", "Explainability Requirements", "Audit Trail Content", "Evidence Retention",
        "Human-in-the-Loop Controls", "Automated Decision Limits", "Override Governance", "Four-Eyes Review",
        "Regulatory Examination Readiness", "Internal Audit Interface", "Management Information", "Key Risk Indicators",
        "Issue Tracking", "Remediation Standards", "Vendor Risk", "Business Continuity", "Incident Response",
        "Staff Vetting", "Whistleblowing", "Conflicts of Interest", "Gifts and Hospitality", "Training Curriculum",
        "Testing Programme", "Lookback Reviews", "Jurisdiction Risk Lists", "Correspondent Relationships",
        "Policy Exceptions", "Handbook Governance"]),
    80: ("Enterprise Credit, KYC and Compliance Risk Framework", "ER", [
        "Risk Appetite Statement", "Governance Structure", "Roles and Responsibilities", "Risk Taxonomy",
        "Credit Risk Principles", "Retail Lending Standards", "DTI and Affordability Limits", "Score Cut-Offs",
        "Thin-File Treatment", "Secured Lending and LTV", "Pricing Discipline", "Underwriting Authority Matrix",
        "Exceptions Framework", "Portfolio Monitoring", "Early Warning Indicators", "Collections Strategy",
        "Provisioning Principles", "Stress Testing", "Concentration Risk", "Model Risk Management",
        "KYC Framework Overview", "Customer Risk Scoring", "Risk Tier Definitions", "EDD Standards",
        "PEP Framework", "Sanctions Framework", "Watchlist Management", "Screening Calibration",
        "Graph Analytics Governance", "Relationship Hop Limits", "Shared Identifier Policy", "Fraud Ring Detection",
        "Confirmed Fraud Handling", "Escalation Matrix", "Compliance Case Handling", "SAR Governance",
        "Customer Communication Rules", "Adverse Action and Disclosures", "Fair Treatment", "Complaints Handling",
        "Data Governance", "Data Lineage", "Data Quality Metrics", "Privacy Impact Assessments",
        "Information Security Baseline", "Access Control", "Third-Party Risk", "Outsourcing Standards",
        "AI and LLM Use Policy", "Prompt and Output Controls", "Hallucination Risk Controls", "Human Oversight of AI",
        "Automated Decisioning Limits", "Retrieval Source Governance", "Audit Trails for AI Decisions", "AI Incident Response",
        "Operational Resilience", "Business Continuity Planning", "Change Management", "Release Controls",
        "Testing and Validation", "Independent Review", "Internal Audit Coordination", "Regulatory Engagement",
        "Reporting and MI", "KPI Definitions", "KRI Thresholds", "Board Reporting",
        "Training and Culture", "Conduct Standards", "Remuneration and Incentives", "Whistleblowing",
        "Records Management", "Retention Schedule", "Legal Holds", "Document Control",
        "Framework Exceptions", "Framework Review Cycle", "Glossary of Terms", "Appendix of Control Mappings"]),
}

SUBJ = ["The Bank", "Underwriting staff", "Compliance analysts", "The automated decision system", "Relationship managers",
        "The credit risk function", "Operations teams", "Senior management", "The KYC team", "Quality assurance reviewers"]
OBL = ["must", "shall", "is required to", "must not", "may only"]
ACT = ["document the evidence relied upon for every decision", "retain a complete audit trail of the data, rules and evidence used",
       "escalate cases that meet the defined risk criteria within the required service level",
       "apply the thresholds in this policy consistently across all channels",
       "record any exception with a named approver and a written rationale",
       "re-verify identity data when material inconsistencies are detected",
       "obtain additional documentation before progressing the application",
       "refer the case for manual review rather than auto-approving",
       "avoid disclosing the reason for a compliance escalation to the applicant",
       "cross-check applicant relationships against the knowledge graph before final decision",
       "cite the specific policy clause supporting each outcome",
       "reassess the customer risk rating when new relationship information emerges"]
COND = ["where the applicant presents a thin credit file", "when an identifier is shared with another applicant",
        "if any related party carries a confirmed fraud flag", "when the debt-to-income ratio approaches the limit",
        "if the applicant is, or is related to, a politically exposed person",
        "when a sanctions name match is returned with a corroborating date of birth",
        "in the event of a recent address change", "for applications above delegated authority limits",
        "when source-of-funds documentation is incomplete", "during periodic and trigger-event reviews",
        "when model outputs conflict with deterministic rules", "for self-employed or variable-income applicants"]
RATIONALE = ["This control reduces the likelihood of undetected fraud and supports regulatory examination readiness.",
             "The requirement ensures consistent treatment of comparable customers and defensible outcomes.",
             "Evidence from internal reviews shows this safeguard materially lowers loss and compliance incidents.",
             "This provision aligns the operating model with the risk appetite approved by the Board.",
             "Consistent application allows management information to be compared across periods and teams."]

# Anchor clauses repeated in the relevant docs so retrieval has concrete, consistent facts.
ANCHORS = {
    "LP": ["Debt-to-income ratio above 43% must be declined; between 36% and 43% requires a credit score of 700 or higher.",
           "Applicants with less than 2 years of credit history are thin-file and must be referred to manual underwriting.",
           "A credit score below 580 must be declined; scores of 580 to 679 are referred to manual underwriting.",
           "Secured loans with loan-to-value above 90% are referred and above 100% are declined."],
    "KP": ["HIGH risk applies when an identifier is shared with a party carrying a confirmed fraud flag; such cases are escalated to Compliance.",
           "A PEP or a close family member of a PEP requires enhanced due diligence and cannot be auto-approved.",
           "Two or more unrelated applicants sharing a phone number or address are a potential fraud ring indicator.",
           "MEDIUM risk cases always go to manual review with source-of-funds and source-of-wealth evidence."],
    "AO": ["A name similarity of 90% or more combined with a matching date of birth is a potential sanctions match and the case is frozen and escalated.",
           "Automated systems may recommend but never finalize a sanctions-based decline without human confirmation.",
           "Compliance analysts must review escalated cases within 2 business days.",
           "Applicants must never be told that an application was escalated for sanctions or suspicion (no tipping off)."],
    "ER": ["HIGH KYC risk cases must not be auto-approved; the LLM may raise but never lower a rule-based risk floor.",
           "Every automated decision must cite policy clauses and graph evidence and retain an audit trail.",
           "Deterministic credit rules take precedence over model recommendations in all cases.",
           "Medium-risk KYC outcomes are referred to manual review even when credit metrics support approval."],
}


def page_clauses(rng: random.Random, prefix: str, ch: int, anchor: str | None) -> list[tuple[str, str]]:
    out = []
    n = 5
    for i in range(1, n + 1):
        if i == 1 and anchor:
            text = anchor
        else:
            text = (f"{rng.choice(SUBJ)} {rng.choice(OBL)} {rng.choice(ACT)} {rng.choice(COND)}. "
                    f"{rng.choice(RATIONALE)}")
        out.append((f"{prefix}-{ch}.{i}", text))
    return out


def build_pdf(pages: int, title: str, prefix: str, topics: list[str]):
    rng = random.Random(pages)
    topics = (topics + [f"Supplementary Controls {n}" for n in range(1, pages)])[:pages - 1]  # 1 cover + N-1 sections
    path = OUT / f"{prefix.lower()}_{pages}pages_{title.lower().replace(',', '').replace(' ', '_')}.pdf"
    doc = SimpleDocTemplate(str(path), pagesize=A4, title=title, author="Synthetic Bank (demo)",
                            leftMargin=56, rightMargin=56, topMargin=56, bottomMargin=56)
    story = [Spacer(1, 200), Paragraph(title, H1), Spacer(1, 12),
             Paragraph(f"Document {prefix}  |  Version 1.0  |  {pages} pages", H2), Spacer(1, 8),
             Paragraph("Synthetic policy document for demonstration of the loan processing and KYC triage "
                       "Graph RAG system. Not real regulatory guidance.", BODY)]
    anchors = itertools.cycle(ANCHORS[prefix])
    spread = {i for i in range(0, len(topics), max(1, len(topics) // 12))}
    for ch, topic in enumerate(topics, start=1):
        story.append(PageBreak())
        story.append(Paragraph(f"Section {ch}. {topic}", H1))
        story.append(Paragraph(f"Applies to: all retail loan applications and KYC triage activity. Reference {prefix}-{ch}.", H2))
        story.append(Spacer(1, 6))
        anchor = next(anchors) if (ch - 1) in spread else None
        for cid, text in page_clauses(rng, prefix, ch, anchor):
            story.append(Paragraph(f"<b>{cid}</b>  {text}", BODY))
    doc.build(story, onFirstPage=_footer(title), onLaterPages=_footer(title))
    return path


def _footer(title):
    def f(canvas, doc):
        canvas.setFont("Helvetica", 8)
        canvas.drawString(56, 30, f"{title} - Synthetic demo document")
        canvas.drawRightString(A4[0] - 56, 30, f"Page {doc.page}")
    return f


# ---- applications -----------------------------------------------------------
def app(i, name, dob, inc, debt, loan, term, score, hist, pv, notes, pid=None):
    return {"app_id": f"APP-{i}", "person_id": pid or f"NEW-APP-{i}", "applicant_name": name, "dob": dob,
            "monthly_income": inc, "existing_monthly_debt": debt, "loan_amount": loan, "loan_term_months": term,
            "credit_score": score, "credit_history_years": hist, "property_value": pv, "notes": notes}


BATCH_JSON = [  # mixes existing graph people (known relationships) with new names
    app(101, "Rohan Mehta", "1990-05-19", 11000, 900, 120000, 240, 745, 10, 260000, "Software engineer, 6 years at current employer. Home renovation.", None),
    app(102, "Lucia Fernandez", "1996-12-02", 4200, 350, 18000, 48, 655, 3, None, "Teacher, moderate credit score after medical debt."),
    app(103, "Elena Cruz", "1994-02-17", 5200, 450, 30000, 60, 715, 5, None, "Existing customer applying for a personal loan.", "P-006"),
    app(104, "Victor Ruiz", "1990-11-05", 5800, 600, 25000, 48, 700, 7, None, "Returning applicant; previous loan closed.", "P-005"),
    app(105, "Hannah Weiss", "1984-07-23", 7000, 3000, 80000, 120, 690, 14, None, "High existing obligations; consolidation loan."),
    app(106, "Ivan Petrovic Sokolov", "1975-03-12", 12500, 1000, 250000, 180, 770, 18, 380000, "Business owner, import and export."),
    app(107, "Mei Lin Zhao", "1999-03-30", 3800, 200, 12000, 36, 720, 1, None, "Recent graduate, one credit card for 10 months."),
    app(108, "Omar Haddad", "1978-10-09", 9800, 700, 95000, 180, 810, 22, 140000, "Long-standing customer, high LTV request."),
]

CSV_ROWS = [  # new people; mix of clean, thin-file, PEP-name, low-score, high-DTI, high-LTV
    app(201, "Sofia Almeida", "1987-01-11", 8200, 650, 60000, 84, 735, 11, None, "Nurse practitioner. Vehicle purchase."),
    app(202, "Karim Rahman", "1961-12-01", 15000, 1500, 200000, 180, 790, 25, 320000, "Former government official, wealth from investments."),
    app(203, "Daniel Okafor", "1993-04-15", 3500, 1300, 40000, 60, 640, 4, None, "Gig worker with variable income and credit card balances."),
    app(204, "Grace Thompson", "1982-09-27", 6900, 500, 55000, 72, 570, 9, None, "Recent missed payments; score has dropped."),
    app(205, "Jun Park", "1991-06-06", 7600, 400, 45000, 60, 765, 8, None, "Clean history. Education financing."),
    app(206, "Fatima Noor", "2002-08-18", 3000, 150, 9000, 24, 700, 1, None, "Student with part-time job, no co-signer."),
    app(207, "Leo Marino", "1980-02-02", 10500, 800, 330000, 300, 740, 13, 340000, "Cash-out refinance; very high loan to value."),
    app(208, "Ivan Sokolov", "1975-03-12", 9000, 700, 70000, 96, 760, 16, None, "Trader; same date of birth as a previously flagged name variant."),
]


def main():
    for pages, (title, prefix, topics) in DOCS.items():
        p = build_pdf(pages, title, prefix, topics)
        print(f"{p.name}: {p.stat().st_size / 1024:.0f} KiB")
    (OUT / "applications_batch_A.json").write_text(json.dumps(BATCH_JSON, indent=2), encoding="utf-8")
    with (OUT / "applications_batch_B.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(CSV_ROWS[0]))
        w.writeheader()
        w.writerows(CSV_ROWS)
    print("applications_batch_A.json, applications_batch_B.csv")


if __name__ == "__main__":
    main()
