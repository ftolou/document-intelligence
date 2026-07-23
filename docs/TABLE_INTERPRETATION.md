# Dedicated Table Interpretation Stage

Phase 2.1 adds a dedicated LLM step between VLM/OCR evidence and the main receipt parser.

## Why

The pipeline already passed structured VLM table rows to the main parser, but the main parser had to do too many tasks in one prompt: merchant/date extraction, item extraction, table interpretation, totals, payments, validation-related reasoning, and JSON formatting.

For receipts with clear VLM tables, this could still lose structure. Example failure mode:

```text
ELVITAL SHAMPOO Coupon SORTIMENT | 2,39 | -0,24 | 2,15a
```

A one-pass parser may flatten this into the product name and later categorize the product as a coupon. The dedicated table interpreter first separates:

```json
{
  "description": "ELVITAL SHAMPOO",
  "original_price": 2.39,
  "discount_amount": -0.24,
  "line_total": 2.15,
  "tax_code": "a"
}
```

## Design principle

This is not a deterministic receipt-specific cleanup layer.

The deterministic code only selects/truncates evidence, calls the LLM, stores artifacts, and validates the JSON wrapper. The LLM infers the table semantics from:

- repeated row layout
- cell order
- right-aligned numeric columns
- amount signs
- neighbouring rows
- tax suffixes such as `a` / `b`
- total/payment/change wording
- arithmetic consistency
- headers when present, but headers are not required

## New artifacts

For every run with usable structured VLM table evidence, the pipeline writes:

```text
<run_id>_v14_15_table_interpretation.json
<run_id>_v14_15_table_interpretation_prompt.txt
<run_id>_v14_15_table_interpretation_raw.txt
latest_v14_15_table_interpretation.json
latest_v14_15_table_interpretation_prompt.txt
latest_v14_15_table_interpretation_raw.txt
```

The compact interpretation is attached to `visual_evidence["table_interpretation"]` and therefore appears inside the main LLM prompt as high-priority evidence.

## Output schema

The table interpreter returns:

```json
{
  "schema_version": "v14_15_table_interpretation_1",
  "status": "ok",
  "tables": [
    {
      "source_table_id": "vlm_table_00",
      "table_type": "headerless_item_table",
      "column_roles": [
        {"column_index": 0, "role": "description"},
        {"column_index": 1, "role": "unit_or_original_price"},
        {"column_index": 2, "role": "discount_amount"},
        {"column_index": 3, "role": "line_total"}
      ],
      "rows": [
        {
          "source_row_id": "vlm_table_00_row_001",
          "row_type": "item",
          "raw_cells": ["1 ELVITAL SHAMPOO Coupon SORTIMENT", "2,39", "-0,24", "2,15a"],
          "canonical": {
            "description": "ELVITAL SHAMPOO",
            "quantity": 1,
            "original_price": 2.39,
            "discount_amount": -0.24,
            "line_total": 2.15,
            "tax_code": "a"
          },
          "confidence": 0.86
        }
      ]
    }
  ],
  "settlement": {
    "amount_due": 38.02,
    "payments": [{"method": "cash", "amount": 50.0}],
    "change": 12.61,
    "confidence": 0.85
  },
  "warnings": [],
  "overall_confidence": 0.86
}
```

## How it affects the final parser

The main parser now receives this additional instruction:

> If dedicated table interpretation is present, use it as high-priority item-table evidence because it separates columns before final receipt assembly.

The final receipt schema still remains compatible. Extra item fields such as `original_price`, `discount_amount`, `tax_code`, and `table_interpretation_source_row_id` are preserved by normalization and can be displayed in Human Review or imported later.

## Failure behavior

The stage is non-fatal. If the LLM table interpreter fails or no structured table evidence exists, the pipeline continues with the previous VLM/OCR evidence path.


## Phase 2.2 refinement

See `docs/archive/historical-patch-notes/archive/historical-patch-notes/TABLE_INTERPRETATION_REFINEMENTS.md` for the product-description/line-note separation, settlement/tender refinement, and printed-change validation added in `v1.6-phase2-table-refinement`.
