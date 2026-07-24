# Extraction state contracts

The extraction workflow exchanges typed, stage-owned artifact groups rather than storing every intermediate value directly on a generic mutable context.

The lifecycle is:

```text
created -> prepared -> visual_ready -> parsed -> repaired -> finalized
```

Each default stage declares its required input phase and produced output phase. The workflow validates the phase before running the stage and advances it only after successful completion.

Artifact ownership is grouped as follows:

- `PreparedArtifacts`: output paths and preliminary OCR context.
- `VisualArtifacts`: VLM output, crop re-OCR evidence, table interpretation, and arbitration.
- `ParsingArtifacts`: main LLM result, OCR context, receipt, compact evidence, and initial validation.
- `RepairArtifacts`: recovery attempts and the currently selected receipt/report candidate.
- `FinalizationArtifacts`: final receipt representation, categorization result, and pipeline metadata.

`ExtractionContext` remains the runtime envelope for configuration, injected dependencies, progress reporting, and observability. It no longer exposes the string-based `require("field")` data-bus API. Typed accessors raise `StageContractError` when a stage attempts to consume an artifact that has not been produced.

The static contract check is executed with:

```bash
python scripts/check_extraction_state_boundaries.py
```
