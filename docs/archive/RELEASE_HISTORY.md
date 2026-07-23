# Historical Release Notes

The sections below were moved from the project README during the Phase 1 maintainability cleanup. They preserve the evolution of the extraction pipeline but are not current operating instructions.

### Phase 2.1: Dedicated table interpretation

The pipeline now includes an optional dedicated LLM table-interpretation stage before the main receipt parser. It uses structured VLM/OCR table evidence to infer headerless item-table semantics: product description, quantity, original/unit price, item-level discount, final line total, tax marker, settlement rows and payment/change rows. This avoids flattening contextual cells such as coupon/discount notes into the product name. See `docs/TABLE_INTERPRETATION.md`.


## Phase 2.2 refinement

See `docs/archive/historical-patch-notes/TABLE_INTERPRETATION_REFINEMENTS.md` for the product-description/line-note separation, settlement/tender refinement, and printed-change validation added in `v1.6-phase2-table-refinement`.

### Phase 2.3: compact table interpretation + external prompt files

This package adds a compact dedicated table-interpretation prompt and moves the main LLM prompts into separate template files under `src/receipt_intelligence/prompts/`. The table interpreter now returns a smaller canonical artifact to reduce truncated JSON failures. It also strengthens generic guidance for headerless tables, quantity/unit-price rows, product text vs promotion notes, `%` inside product names, and settlement/tender rows. See `docs/PROMPT_FILES.md` and `docs/archive/historical-patch-notes/TABLE_INTERPRETATION_REFINEMENTS.md`.


## Phase 2.3: table arbitration + patch correction

Version `v1.8-phase2-table-arbitration-patch` adds a dedicated OCR/VLM arbitration layer and a compact patch-style correction pass. The new layer flags shifted VLM tables, exposes OCR layout item/price candidates, links quantity/unit-price notes to product rows, and prevents product percentages such as `MOZZARELLA 40%` from becoming tax evidence without explicit tax context. See `docs/archive/historical-patch-notes/TABLE_ARBITRATION_AND_PATCH_CORRECTION.md`.


### Phase 2.3 update — Authoritative table assembly + VLM CLI mode

This package switches the VLM service runner to `cli` mode by default and adds authoritative table assembly. The dedicated table interpretation result is now used as the item-table source, so the main parser does not need to re-extract every item from a huge prompt. If the main parser fails but the table interpretation is usable, the pipeline creates a provisional `needs_review` receipt instead of an empty `llm_failed` result. See `docs/archive/historical-patch-notes/AUTHORITATIVE_TABLE_ASSEMBLY_AND_VLM_CLI.md`.


### Phase 2.3 row-level table arbitration

The receipt table assembler now fuses OCR and VLM evidence per row instead of choosing one source globally. It deduplicates overlapping OCR rows, attaches quantity notes to the correct items, and can add VLM-only or orphan adjustment rows when total reconciliation supports them. See `docs/archive/historical-patch-notes/ROW_LEVEL_TABLE_ARBITRATION.md`.


### Phase 2.6: Batch Review Queue, duplicate detection and DB management

This package adds an operational review workflow for batch processing:

- Batch and single runs are registered in a SQLite `review_queue` table.
- The new **Review Queue** tab lets you filter pending, rejected, duplicate and approved receipts.
- Queue entries can be opened in the existing side-by-side Human Review screen.
- Duplicate candidates are scored using file hash, merchant/date/time/total and item-overlap evidence.
- The Ask Your Receipts tab now includes DB management controls to delete one approved receipt or delete all approved/imported receipt records.

Review queue entries are staging data. Only approved/imported receipts enter the trusted receipt/item database used by Ask Your Receipts. See `docs/archive/historical-patch-notes/BATCH_REVIEW_QUEUE_AND_DUPLICATES.md`.

## Phase v1.13 — Vertical price-stack recovery

This phase adds a validation-gated recovery layer for receipts where right-side prices are visible as a vertical column but OCR/layout reconstruction split or missed them. The recovery crops the whole item-price column, OCRs original and 2x-upscaled variants, pairs amount rows with left-side product rows, and applies the candidate only if printed-total validation improves. Balanced receipts are skipped and recovered rows require human review before DB/RAG import.

The final/reconciled receipt artifacts now also preserve item categorization metadata so the Review Queue and database import do not lose `category_key`, `category_group`, or category confidence fields.


## Phase v1.13.1 — Vertical price-stack parser fix

This patch strengthens the vertical price-stack recovery layer before adding the Query Planner. The crop itself was already useful, but narrow-column OCR sometimes returned only a few clean amount strings. The recovery now creates multiple OCR-preprocessed crop variants, clusters OCR boxes by y-position, parses noisy price strings more tolerantly, uses full-image right-column OCR as fallback evidence, and selects the evidence source that best improves printed-total validation.

Modified file:

```text
src/receipt_intelligence/pipeline/receipt_vertical_price_stack_recovery_v14.py
```

No VLM rebuild is required. Restart only the app container after copying the package.


### v1.13.2 vertical stack balanced-fusion fix

This patch tightens the vertical price-stack recovery layer:

- partial improvements are now candidate-only and are not written into the final receipt;
- digit-only OCR parsing no longer creates false prices when a decimal amount is already present in the same OCR row;
- fused region re-OCR + table-arbitration evidence is evaluated before crop OCR;
- single unmatched product rows can receive the exact printed-total residual only when the fused evidence otherwise balances;
- all applied recoveries remain `needs_review` and `requires_review=true`.

### v1.13.3 patch-only correction

- Removes the full-receipt LLM correction fallback that could truncate large JSON outputs.
- Uses only compact validated patch operations.
- Retries malformed patch JSON once with a stricter output budget.
- Keeps the original receipt unchanged when a patch fails or does not improve deterministic validation.
- See `docs/archive/historical-patch-notes/PATCH_ONLY_CORRECTION.md`.


## v1.22.0 — Phase 7 observability and regression hardening

- Adds extraction and query telemetry, readiness checks, and stable test profiles.
- Adds a deterministic natural-language query regression corpus.
- Defines evidence-based removal gates for the retained query fallback.

## v1.23.0 — Phase 8 single query engine cutover

- Removes the v1 one-shot query engine and all query compatibility aliases.
- Makes LangGraph the only query execution path.
- Moves deterministic fallback into the v2 planner so validation and tools remain shared.
- Makes safe tools call the parameterized receipt database directly.

## v1.21.0 — Phase 6 legacy cleanup

- Completes the runtime cutover to the single `var/` root.
- Removes legacy runtime mounts, fallback reads, and automatic database copying.
- Removes the deprecated `receipt_intelligence.qa` import alias package.
- Removes superseded Docker helper wrappers and archives completed phase docs.
- Retains the active `query/legacy` fallback and active `_v14` extraction algorithms.

## v1.24.0 — Phase 9 extraction module cleanup

- Moves active receipt algorithms from release-numbered `pipeline/*_v14.py` files into responsibility-based `extraction/` packages.
- Leaves `pipeline/integrated_receipt_pipeline.py` as the stable public entry point.
- Removes the unused full-receipt correction module and prompt.
- Renames the active batch runner scripts without changing persisted receipt schema or artifact compatibility.
