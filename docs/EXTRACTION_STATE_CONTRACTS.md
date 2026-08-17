# Extraction state contracts

The extraction workflow exchanges typed, stage-owned artifact groups rather than storing
intermediate values on a generic mutable context.

The lifecycle is:

```text
created -> prepared -> transcribed -> extracted -> validated -> corrected -> categorized -> finalized
```

Each stage declares its required input phase and produced output phase. The workflow validates
the phase before running the stage and advances it only after successful completion.

Artifact ownership is grouped as follows:

- `PreparedArtifacts`: managed output paths and source-image metadata.
- `TranscriptionArtifacts`: Paddle geometry, crop plan, and canonical Qwen transcription.
- `StructuredArtifacts`: Gemma scalar/item extraction and the normalized receipt candidate.
- `ValidationArtifacts`: read-only deterministic findings and routing signals.
- `CorrectionArtifacts`: accepted bounded corrections and the corrected receipt candidate.
- `FinalizationArtifacts`: categorization data, stable receipt representation, and metadata.

`ExtractionContext` is the runtime envelope for configuration, injected dependencies, progress
reporting, and observability. Typed accessors raise `StageContractError` when a stage attempts to
consume an artifact that has not been produced.

Run `python scripts/check_extraction_state_boundaries.py` to enforce the static contract.
