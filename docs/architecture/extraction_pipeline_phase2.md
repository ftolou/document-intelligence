# Extraction pipeline migration — Phase 2

Phase 2 imports the proven Paddle-geometry/Qwen transcription subsystem behind the Phase 1
contracts. It remains inactive in the production workflow factory.

## Added behavior

- PaddleOCR supplies detection polygons only; recognized Paddle text is ignored.
- Detected boxes are clustered into approximate physical receipt rows.
- Crop count adapts to detected-row density and image height/width aspect ratio.
- Nominal crop boundaries are snapped to Paddle-proposed gaps and verified against source-image
  pixel ink density.
- Unsafe crop plans fall back to one full-image Qwen call.
- Qwen crop calls may run in parallel and retry only transport/call failures.
- When any planned crop fails, every partial crop transcription is discarded and the whole image
  is retried.
- Successful Qwen output receives only transport cleanup and ordered concatenation. No semantic,
  duplication, line-count, arithmetic, or protocol acceptance validator is added.
- Global `R0001...` row IDs produce one canonical evidence representation for later Gemma stages.

## Production safety

`build_default_extraction_workflow()` is intentionally unchanged. The existing production flow
continues to run `PreparationStage -> VisualEvidenceStage -> SpatialOverviewStage ->
MainParsingStage -> RepairAndCorrectionStage -> FinalizationStage`.

The new `TranscriptionStage` is exported but not selected. It can be exercised through
`scripts/run_next_transcription.py` or an explicitly constructed test workflow.

## Deliberately excluded

- Gemma scalar extraction
- Gemma item extraction
- receipt assembly
- deterministic receipt validation
- correction coordinator integration
- replacement of `JobProcessingService` OCR ownership
- production factory changes

Those belong to later phases.
