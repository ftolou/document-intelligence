# Phase 7.2 — Next-pipeline integration hardening

Phase 7.2 fixes integration defects observed in the first receipt run after prompt delivery was restored.
It does not change the active strategy selector or remove the current workflow rollback path.

## Changes

- Adapts next-pipeline item fields (`name`, `final_price`) to the legacy categorizer input contract on a deep copy.
- Preserves the category-only overlay, so categorization cannot mutate receipt arithmetic.
- Normalizes schema-nullable model placeholders such as `?`, `null`, and `unknown` to JSON null.
- Canonicalizes valid three-letter currency values and rejects invalid currency strings.
- Enforces printed-only receipt-level discount totals; no discount is calculated in Python.
- Enables receipt number, net amount, payment method, and payment received scalar tasks.
- Routes `ITEM_DISCOUNT_ARITHMETIC` to the bounded item source-evidence specialist.
- Adds item source-evidence prompt/schema version 1.1.0 with explicit original-price and discount evidence.
- Expands the item specialist's permitted patch scope to `final_price`, `original_price`, and `discount_amount` only.
- Marks `next_finalize` as complete in the pipeline metadata snapshot instead of persisting a stale `running` status.

## Safety invariants

- Qwen transcription remains the canonical evidence source.
- Python does not infer or calculate missing monetary values.
- Item corrections require literal source-row evidence and validator acceptance.
- The correction coordinator still has no generic fallback.
- Unsupported or rejected correction candidates remain unapplied.
- Categorization compatibility fields exist only in the isolated categorizer input copy.
