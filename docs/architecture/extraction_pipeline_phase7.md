# Extraction pipeline Phase 7 — opt-in production integration

Phase 7 connects the Phase 1-6 subsystems through the production extraction entry point without
removing or silently replacing the existing workflow.

## Strategy selection

- `EXTRACTION_STRATEGY=current` is the default and runs the existing OCR/VLM workflow.
- `EXTRACTION_STRATEGY=next` runs the Qwen/Gemma workflow.
- Callers may override the environment by passing `extraction_strategy=` to
  `run_receipt_extraction` or `run_integrated_receipt_pipeline`.
- Unknown values fail closed with `ValueError`.

## Next workflow

1. `NextPreparationStage`
2. `TranscriptionStage`
3. `StructuredExtractionStage`
4. `ValidationStage`
5. `CorrectionStage`
6. `CategorizationStage`
7. `NextFinalizationStage`

The workflow preserves the application result keys and the current final artifact filenames.

## Deliberate boundary

Phase 7 activates the new extraction implementation but does not yet remove the application's
pre-extraction OCR call or the legacy OCR/VLM stages. That cleanup belongs after full end-to-end
regression proves the new path. The current strategy remains an immediate rollback path.
