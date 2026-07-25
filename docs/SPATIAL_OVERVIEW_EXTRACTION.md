# Spatial geometry extraction

The receipt extraction workflow preserves OCR geometry through semantic parsing. It is the
only supported extraction path; there is no legacy/current strategy switch.

## Workflow

```text
OCR words and normalized boxes
    -> optional VLM and region re-OCR hypotheses
    -> canonical spatial document map
    -> deterministic same-band row and column grouping
    -> high-resolution region item-price candidates
    -> schema-constrained geometry-first main receipt parser
    -> deterministic validation
    -> validation-gated field-level price fusion
    -> bounded re-OCR and patch-only correction when useful
```

Geometry processing only preserves observable document structure: reading order, row bands,
x-ranges, amount-column centers and source IDs. Product, discount, loyalty, tax and payment
semantics remain LLM responsibilities.

High-resolution region re-OCR may supplement a damaged or missing full-image line price when
product text, spatial alignment, confidence, and explicit source lines agree. This evidence can
patch only money/quantity fields on an existing matched item; it cannot add, remove, reorder, or
rename items and is selected only when receipt validation improves.

PaddleOCR-VL tables and deterministic OCR/VLM arbitration are secondary hypotheses. They do
not replace the OCR geometry and cannot overwrite the complete item list. The removed
dedicated table-interpreter, authoritative table assembler, right-column item recovery and
vertical-price-stack reconstruction were obsolete because they could trade semantic accuracy
for arithmetic balance.

## Configuration

The geometry map is always enabled. The only spatial formatting setting is:

```env
SPATIAL_CANVAS_WIDTH=112
```

Valid values are 72 through 160. The default is suitable for the main receipt parser.

For the manual runner:

```bash
python scripts/run_receipt_pipeline.py OCR.json \
  --source-image receipt.jpg \
  --enable-vlm
```

## Evidence hierarchy

1. Full-image OCR geometry and original line IDs for page structure.
2. Deterministic same-band row and column grouping derived from that geometry.
3. High-confidence region crop OCR for supplemental product/line-price evidence.
4. VLM tables and arbitration as fallible hypotheses.
5. Main-LLM semantic interpretation constrained to the canonical receipt JSON Schema.
6. Arithmetic reconciliation as validation, not proof of row identity.

The parser is instructed not to convert position or article-number columns into quantities and
not to use loyalty rows as product descriptions.

## Artifacts

Each run writes:

- `<run_id>_spatial_document_map.json`
- `<run_id>_spatial_canvas.txt`
- `<run_id>_spatial_overview.json`
- `<run_id>_line_price_fusion.json`
- `<run_id>_receipt_line_price_fused.json`
- `<run_id>_validation_report_line_price_fused.json`

`spatial_overview.json` is deterministic geometry metadata. No separate overview model call is
made. The model-call dashboard records `receipt_main_parse_spatial` for extraction.

## Evaluation criteria

Regression evaluation should compare description-to-price association, quantity accuracy,
discount row type, tax-field accuracy, human-review rate and false automatic imports. A
balanced item sum alone is not sufficient.
