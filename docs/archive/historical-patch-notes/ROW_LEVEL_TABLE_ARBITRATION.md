# Phase 2.3 — Row-level OCR/VLM table arbitration

This release changes the table assembly behavior from **global source selection** to **row-level fusion**.

Earlier, when the VLM table showed a row-shift risk, the assembler preferred OCR layout rows globally. That fixed shifted top rows, but could lose later rows where the VLM table was more complete. The new behavior combines the two sources per row.

## What changed

1. **OCR layout rows remain the base** when a VLM row shift is detected.
2. **Overlapping duplicate OCR rows are removed**, for example duplicate `PFAND 0,48` rows sharing the same OCR line and amount.
3. **Quantity/unit-price notes are attached** to the matching product or deposit row, for example:
   - `2 Stk x 1,39` → `CHIPSFRISCH`, quantity `2`, unit price `1.39`
   - `2 Stk x 3,39` → `PISTAZIEN`, quantity `2`, unit price `3.39`
   - `3 Stk x 0,25` → `LEERGUT EINWEG`, quantity `3`, unit price `0.25`
4. **VLM-only item rows can be added back** when OCR missed them and total reconciliation supports them.
5. **Negative orphan adjustment rows** can be recovered when they reconcile the printed total.
6. The table assembly report now includes the reconciliation decision, selected supplements, dedupe actions, and quantity-note attachments.

## Why this is still general

This is not a REWE-specific rule. The logic uses generic receipt evidence:

- duplicate overlap by OCR source line and amount;
- row-shift warnings from VLM/OCR disagreement;
- quantity-note arithmetic;
- total reconciliation against the printed total;
- deposit/refund vocabulary for classification only.

The LLM table interpreter remains the semantic table source. The deterministic layer only arbitrates between already extracted evidence and validates arithmetic consistency.

## New/updated artifact behavior

The existing artifact remains:

```text
latest_v14_19_table_assembly_report.json
```

But the `item_source.source` field can now be:

```text
table_arbitration_row_level_hybrid
```

The report includes fields such as:

```json
{
  "dedupe_actions": [],
  "quantity_note_actions": [],
  "selected_supplements": [],
  "reconciliation": {
    "status": "matched",
    "base_sum": 29.58,
    "target_total": 32.92,
    "selected_sum": 3.34,
    "residual_error": 0.0
  }
}
```

## Deposit/refund categorization

The category taxonomy now separates:

- `deposit_pfand`: positive bottle/can deposits;
- `deposit_refund`: negative Leergut/Pfand return rows;
- `discount_coupon`: coupons, rebates and promotions.

This improves DB/RAG quality for questions such as:

- “How much did I pay in deposits?”
- “How much Leergut refund did I get?”
- “Which purchases had coupons?”
