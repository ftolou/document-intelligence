# Right-column price recovery

Phase v1.12 adds a guarded recovery layer for receipts where the left-side item
text is visible but the right-side price column was missed or attached to the
wrong product row.

## Why this exists

Some receipts are readable to a human but still fail because OCR splits rows like:

```text
KART. VORW. FESTK.
1,99 B
EIER FH. G.M-L
2,99 B
```

or because the price is recognized as a separate line. The LLM cannot reliably
recover the item table when the structured evidence has already lost the
item/price pairing.

## Safety rule

Right-column recovery is **not** a new default parser. It only runs after
validation reports an item-total problem.

Balanced receipts are left untouched:

```text
if extracted item sum == printed total:
    skip right-column recovery
```

Recovered rows are marked as `requires_review=true` and should not be trusted for
analytics/RAG import until a human approves them.

## Pipeline position

```text
OCR / VLM / table interpretation
→ row-level OCR/VLM arbitration
→ main parser
→ validation
→ bounded right-column re-OCR
→ right-column recovery candidates
→ validation-gated add/replace patches
→ human review
```

## Artifacts

The pipeline writes these artifacts when the recovery layer is reached:

```text
latest_v14_21_right_column_recovery.json
latest_v14_21_receipt_right_column_recovered.json
latest_v14_21_validation_report_right_column_recovered.json
```

## What it can do

- Replace a shifted item label when the same OCR lines/amount clearly point to a
different product label.
- Add a recovered product row only if the bounded right-column OCR evidence gives
a candidate price and adding it improves printed-total reconciliation.
- Select the smallest candidate subset that explains the item-total difference.

## What it should not do

- It should not rebuild all items from scratch.
- It should not modify already balanced receipts.
- It should not import recovered rows into DB/RAG without review.
- It should not treat unit-price rows as standalone products unless the final
reconciliation proves they are actual missing item totals.
