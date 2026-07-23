# Portfolio Phase 2 — Receipt Intelligence Database + Hybrid RAG Search

Phase 2 turns the receipt parser into a small **AI Receipt Intelligence** system.
The goal is to demonstrate that AI output can become reusable business data, not
only one-off JSON extraction.

## What Phase 2 adds

- Local SQLite database under `data/receipt_intelligence.db`
- Receipt-level table: merchant, date, totals, review status, artifact paths
- Item-level table: raw item name, normalized/category fields, quantity, price,
  VAT/tax fields, confidence, enriched retrieval text
- Lightweight local retrieval index using SQLite FTS5 when available
- Deterministic lexical fallback when FTS5 is not available
- Natural-language UI: **Ask Your Receipts**
- Schema-constrained local LLM Query Planner with deterministic fallback
- Safe operations for item search, receipt search, totals, counts, averages, and groupings
- APIs for database summary, manual import, and natural-language search
- Backfill script for existing `approved_receipt.json` artifacts

## Correct AI architecture

Phase 2 deliberately separates responsibilities:

```text
LLM/OCR pipeline        -> extracts receipt facts
Human review           -> approves/corrects important fields
SQLite database        -> source of truth for receipt and item facts
RAG/retrieval index     -> finds semantically related items/receipts
Python/SQL             -> performs exact calculations
LLM-ready answer layer  -> formats or later explains the result
```

The system does **not** ask an LLM to calculate financial totals from raw chunks.
For portfolio credibility, this is important: exact facts and sums remain
deterministic.

## Example questions

```text
Which receipt had the shampoo I bought at dm?
Show me the receipts where I bought hygiene products.
How much did I spend on shampoo?
Show receipts with baby products.
Find purchases related to cleaning supplies.
```

The first two questions demonstrate RAG-style retrieval because receipt item text
may contain names such as `HEAD&SHOULDERS`, `ELVITAL`, `FRUCTIS`, `DOVE`, or
`ELMEX` rather than the generic word `shampoo` or `hygiene`.

## APIs

### Database summary

```http
GET /api/receipt-db/summary
```

Returns receipt count, item count, top merchants, top categories, and DB path.

### Manual import

```http
POST /api/receipts/import/<job_id>
```

Imports an existing `approved_receipt.json` artifact for a job. If the approved
artifact does not exist, the endpoint can import the final receipt JSON for local
testing.

Human review already imports automatically.

### Ask receipts

```http
POST /api/ask-receipts
Content-Type: application/json

{
  "question": "Which receipt had the shampoo I bought at dm?",
  "limit": 25
}
```

Response includes:

- generated answer
- parsed query intent
- SQL/filter interpretation
- matched item evidence
- retrieval method used
- source-of-truth note

## Backfill old reviewed receipts

```powershell
python scripts/import_approved_receipts_to_db.py --results-dir outputs/results
```

For testing before human review exists:

```powershell
python scripts/import_approved_receipts_to_db.py --results-dir outputs/results --include-final
```

## Smoke demo without running OCR/Ollama

```powershell
python scripts/demo_receipt_qa.py
```

This creates a temporary in-memory-style demo database, inserts a fake dm receipt,
and asks:

- Which receipt had the shampoo I bought at dm?
- Show me the receipts where I bought hygiene products.
- How much did I spend on shampoo?

## Portfolio message

This phase demonstrates:

- item-level data modeling
- human-in-the-loop approval
- local-first data storage
- RAG-style semantic receipt search
- deterministic financial calculation
- evidence-based answers with traceability

## UI update: separate Ask Your Receipts tab

The Phase 2 natural-language receipt Q&A feature is now separated from the extraction workflow through a dedicated **Ask Your Receipts** tab.

The UI has three top-level workflow tabs:

1. **Run receipt** — single receipt upload, OCR/LLM extraction, human review, and database import.
2. **Batch runner** — serial folder processing for regression and batch testing.
3. **Ask Your Receipts** — hybrid receipt search and spending Q&A over approved receipt items.

This makes the portfolio demo clearer: first approve/import receipts, then switch to the separate Q&A tab to demonstrate RAG-style semantic retrieval and deterministic database analytics.
