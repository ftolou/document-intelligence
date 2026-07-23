# Phase 2.3 — Authoritative Table Assembly and VLM CLI Mode

This version changes two important runtime behaviours.

## 1. VLM service now uses PaddleOCR-VL CLI mode by default

The `receipt-vlm` service now sets:

```yaml
VLM_SERVICE_RUNNER: "cli"
```

The service calls the PaddleOCR `doc_parser` CLI path instead of trying the Python API first. This is intended to avoid the previous Paddle/PaddleOCR-VL Python runtime issues such as undefined `Place` / BF16 support errors.

The image name is intentionally kept stable:

```yaml
image: paddle-gemma-receipt-vlm:gpu-python-cu126
```

Only the runner mode changes. This avoids forcing a heavy VLM image rebuild just because the service now prefers CLI execution.

## 2. Dedicated table interpretation is authoritative for items

Previously the pipeline did:

```text
VLM/OCR evidence
→ dedicated table interpretation
→ huge main parser prompt
→ main parser re-extracts the whole receipt JSON
```

That caused local LLM failures because the main parser prompt became too large and redundant.

The new flow is:

```text
VLM/OCR evidence
→ dedicated table interpretation
→ table arbitration
→ compact main parser prompt
→ authoritative table assembly
→ validation/review
```

When `table_interpretation.status` is `ok` or `partial`, the interpreted rows are treated as the item-table source. The final/main parser no longer needs to re-derive all item rows from raw VLM tables.

If the main parser fails but the table interpretation is usable, the pipeline creates a provisional receipt from the table interpretation instead of returning an empty `llm_failed` receipt.

New artifacts:

```text
latest_v14_19_receipt_table_assembled.json
latest_v14_19_table_assembly_report.json
```

## 3. Table arbitration can override shifted VLM tables

If `latest_v14_18_table_arbitration.json` contains a warning such as:

```text
VLM_TABLE_POSSIBLE_ROW_SHIFT
```

then the assembler prefers OCR layout item/price candidates for item identity and price pairing.

This targets cases where the VLM table is arithmetically plausible but row descriptions and prices are shifted by one visual row.

## 4. Human review remains required

The assembler does not make uncertain receipts automatically importable. Missing merchant/date/payment, payment mismatch, low-confidence rows, or unbalanced totals still produce `needs_review` or `reject`.

The goal is only to avoid losing good table evidence when the main parser fails.
