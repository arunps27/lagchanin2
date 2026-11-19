# Graph RAG: Automated Loan Processing & KYC Triage

LangChain + LangGraph + Groq (`openai/gpt-oss-120b`). All data is synthetic.

## Architecture
```
Policy docs --> Chunk --> Tokenize --> Embed --> FAISS index --> Vector store   (ingest.py)
Customers/phones/addresses/watchlists --> NetworkX knowledge graph               (kg.py)

START -> intake_extract -> graph_retrieve -> policy_retrieve -> kyc_triage
          kyc HIGH  -> human_review (Compliance) -> finalize
          otherwise -> credit_assess -> decision -> REFER? -> human_review -> finalize
```
- **Graph retrieval** finds shared phones/addresses, fraud flags, fuzzy sanctions matches, PEP family links.
- **Policy retrieval** queries the vector store, enriched with the graph findings (graph-first).
- **Guardrails**: the LLM can raise but never lower the deterministic KYC floor, and credit rules (DTI, score, thin file, LTV) and the MEDIUM-KYC-never-auto-approve rule override the LLM.
- `human_review` uses LangGraph `interrupt` + `MemorySaver`; `main.py` auto-confirms unless `--interactive`.

## Setup
```
python -m venv .venv && .venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env        # then set GROQ_API_KEY
python ingest.py              # builds vector_store/
python main.py                # all 6 sample apps
python main.py --app APP-004 --interactive
python main.py --mermaid      # workflow diagram
python -m pytest tests -q     # offline tests (no API key)
```

## Sample outcomes
| App | Scenario | Result |
|---|---|---|
| APP-001 | Clean | APPROVE |
| APP-002 | Thin file | REFER |
| APP-003 | DTI 62% | DECLINE |
| APP-004 | Shares phone/address with fraud-flagged customer | ESCALATE (HIGH KYC) |
| APP-005 | Sanctions near match + DOB | ESCALATE (HIGH KYC) |
| APP-006 | Spouse is a PEP | REFER (MEDIUM KYC, EDD) |
