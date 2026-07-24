# Experimental spatial-overview extraction

The `spatial_overview` strategy tests whether the existing receipt LLM can extract rows more reliably when OCR geometry is preserved instead of flattened into description/amount text.

## Enable it

For the web application, either select **Experimental spatial overview** in the upload form or set:

```env
EXTRACTION_STRATEGY=spatial_overview
```

For the manual runners:

```bash
python scripts/run_receipt_pipeline.py OCR.json \
  --source-image receipt.jpg \
  --enable-vlm \
  --extraction-strategy spatial_overview
```

The default remains `current`, so existing extraction behaviour is preserved unless the strategy is explicitly selected.

## Workflow

```text
OCR words and normalized boxes
    -> optional VLM and region re-OCR hypotheses
    -> canonical spatial document map
    -> deterministic same-band row and column grouping
    -> schema-constrained geometry-first main receipt parser
    -> deterministic validation
    -> patch-only correction when useful
```

The dedicated table-interpreter and separate receipt-overview LLM calls are skipped in this strategy. Generic geometry preserves same-band row membership, x-ranges, amount-column centers, and source IDs. Raw VLM tables and OCR/VLM arbitration remain available as secondary hypotheses for the main semantic parser.

Legacy right-column and full vertical-price-stack item reconstruction are disabled for this strategy because those operations can produce a mathematically balanced but semantically incorrect item list. Patch-only correction remains available.

## Evidence hierarchy

1. Full-image OCR geometry and original line IDs.
2. Deterministic same-band row and column grouping derived from that geometry.
3. Region re-OCR, VLM tables, and arbitration as fallible hypotheses.
4. Main-LLM semantic interpretation constrained to the canonical receipt JSON Schema.
5. Arithmetic reconciliation as validation, not as proof of row identity.

The main parser is explicitly instructed not to convert position/article-number columns into quantities and not to use loyalty rows as product descriptions.

## New artifacts

Each spatial run writes:

- `<run_id>_spatial_document_map.json`
- `<run_id>_spatial_canvas.txt`
- `<run_id>_spatial_overview_prompt.txt`
- `<run_id>_spatial_overview_raw.txt`
- `<run_id>_spatial_overview.json`

No `receipt_spatial_overview` model call is made. The model-call dashboard records only `receipt_main_parse_spatial` for the geometry-first extraction step, making its token count and duration directly comparable with the current strategy.

## Recommended comparison receipts

Use the same OCR/image inputs with both strategies. The most diagnostic cases are:

- a receipt with `Pos`, `Artikel-Nr.`, `Menge`, unit-price, and total columns;
- a receipt with product, loyalty-points, and discount lines interleaved;
- a receipt where two nearby product descriptions were merged;
- a tax table with separate net, tax, and gross columns.

Compare description-to-price association, quantity accuracy, discount row type, tax-field accuracy, review rate, and false automatic imports. A balanced item sum alone is not sufficient.
