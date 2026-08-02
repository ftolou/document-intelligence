# Extraction Pipeline Phase 6: Categorization and Final Publication

Phase 6 separates post-validation presentation responsibilities from extraction, validation, and correction.

## Boundary

```text
CorrectionResult
    -> CategorizationStage
    -> CategorizationResult
    -> NextFinalizationStage
    -> FinalizationResult + compatibility artifacts
```

The active production `factory.py` remains unchanged. Both stages are inactive until the next workflow is activated in a later phase.

## Categorization invariants

- Categorization runs only after correction and final deterministic validation.
- It may append merchant and item category metadata only.
- It must not change descriptions, source rows, quantities, prices, discounts, totals, taxes, payments, currency, or validation status.
- Disabled or failed categorization does not block final publication.
- Existing confidence calibration and review flags remain authoritative.

## Finalization responsibilities

Finalization only:

- attaches the final read-only validation report;
- builds public pipeline metadata;
- writes current UI/review-compatible artifact filenames;
- publishes `latest_*` aliases;
- returns a typed application result.

Finalization performs no semantic correction and no accounting normalization.

## Compatibility artifacts

The Phase 6 filesystem store preserves the currently consumed names, including:

- `<run_id>_receipt_final.json`
- `<run_id>_receipt_final_reconciled.json`
- `<run_id>_receipt_final_categorized.json`
- `<run_id>_v14_validation_report.json`
- `<run_id>_reconciliation_report.json`
- `<run_id>_pipeline_meta.json`
- categorization prompt/raw/result artifacts
- existing `latest_*` aliases

This lets the review queue and artifact service remain unchanged when the next workflow is activated.
