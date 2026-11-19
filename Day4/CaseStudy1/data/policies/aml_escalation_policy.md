# AML and Sanctions Escalation Policy (Synthetic)

## AML-1 Sanctions Screening
AML-1.1 Applicant names and dates of birth are screened against the sanctions watchlist using fuzzy matching.
AML-1.2 A near match (name similarity of 90% or more) combined with a matching date of birth is treated as a confirmed potential match: the application is frozen, declined for auto-processing, and escalated to Compliance.
AML-1.3 A name-only near match with no date of birth match is escalated for analyst review but the applicant is not told of the reason (no tipping off).

## AML-2 Escalation Workflow
AML-2.1 Compliance analysts must review escalated cases within 2 business days.
AML-2.2 Every escalation must record the evidence relied on, including graph relationships and policy clauses.
AML-2.3 Possible suspicious activity is reported through a Suspicious Activity Report (SAR) when the analyst confirms it.

## AML-3 Audit
AML-3.1 Every automated decision must retain an audit trail listing the data, rules and evidence used.
AML-3.2 Automated systems may recommend but never finalize a decline based on sanctions grounds without human confirmation.
