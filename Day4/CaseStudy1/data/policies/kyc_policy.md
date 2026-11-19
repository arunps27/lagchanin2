# KYC Policy (Synthetic)

## KYC-1 Identity Verification
KYC-1.1 Every applicant must present a valid government ID; name and date of birth must match the application.
KYC-1.2 Address must be verified by a utility bill or bank statement issued in the last 90 days.
KYC-1.3 Missing or mismatched identity documents require the application to be referred to manual review.

## KYC-2 Risk Tiering
KYC-2.1 LOW risk: identity verified, no watchlist hits, no shared identifiers with other applicants.
KYC-2.2 MEDIUM risk: politically exposed person (PEP) relationship, recent address change, or thin identity footprint. Enhanced Due Diligence (EDD) is required and the case goes to manual review.
KYC-2.3 HIGH risk: sanctions match, or identifiers (phone, address, bank account, device) shared with an applicant or customer that carries a fraud flag. HIGH risk cases must not be auto-approved and are escalated to Compliance.

## KYC-3 Shared Identifier Rules
KYC-3.1 Two or more unrelated applicants sharing the same phone number or address is a potential synthetic-identity or fraud ring indicator.
KYC-3.2 If any party sharing an identifier has a confirmed fraud flag, the application is HIGH risk and escalated to Compliance.
KYC-3.3 Household members with declared family relationships sharing an address are not by themselves a risk indicator.

## KYC-4 Politically Exposed Persons
KYC-4.1 A PEP or a close family member or associate of a PEP requires EDD: source of funds and source of wealth documentation, and senior management sign-off.
KYC-4.2 PEP-related applications cannot be auto-approved; they are referred to manual review.
