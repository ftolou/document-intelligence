# Phase 2.3: Table Arbitration and Patch Correction

This update adds a stronger evidence layer between OCR/VLM/table interpretation and the main parser.

## Why this exists

Some receipts contain clear OCR row pairing while the structured VLM table is shifted by one row. Example pattern:

```text
OCR layout:  CHICKEN NUGGETZ  -> 2.19
VLM table:   FRISCHKAESE NATU -> 2.19
```

In this case the VLM table is structured but misaligned. The main parser now receives a separate **table arbitration** artifact that cross-checks VLM rows against OCR layout rows.

## New artifact

```text
<run_id>_v14_18_table_arbitration.json
latest_v14_18_table_arbitration.json
```

It contains:

- `ocr_layout_item_candidates`: item/amount pairings from OCR layout rows.
- `quantity_note_candidates`: rows such as `2 Stk x 1,39` linked to nearby product totals.
- `product_percent_not_tax_rows`: product rows such as `MOZZARELLA 40%` that must not become tax rows without explicit tax context.
- `warnings`: e.g. `VLM_TABLE_POSSIBLE_ROW_SHIFT`.

## Correction strategy

A compact patch-style correction pass now runs before the older full JSON correction pass.

New artifacts:

```text
<run_id>_v14_18_correction_patch_prompt.txt
<run_id>_v14_18_correction_patch_raw.txt
<run_id>_v14_18_correction_patch_result.json
<run_id>_v14_18_receipt_patch_corrected.json
<run_id>_v14_18_validation_report_patch_corrected.json
```

The patch pass may return operations such as:

- `replace_field`
- `replace_payments`
- `remove_items`
- `update_item`
- `add_item`

The app applies only a narrow whitelist of operations and keeps the patch result only if deterministic validation improves.

## Generic guards added

- Quantity/unit-price notes are removed from the item list if they leaked as standalone items.
- Product descriptions containing a percent sign no longer become tax evidence unless there is explicit tax/MwSt/USt/VAT/net/gross context.
- Regression reporting now searches direct and nested `job_status.json` files so copied output folders and ZIP extractions are handled more robustly.
