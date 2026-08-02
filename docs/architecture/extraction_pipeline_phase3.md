# Extraction pipeline integration — Phase 3 structured extraction

## Scope

Phase 3 imports the proven Gemma scalar-specialist and direct-item extraction behavior behind
Phase 1 contracts and the Phase 2 canonical transcription result.

Added components:

- provider-neutral schema-constrained chat port and Ollama `/api/chat` adapter;
- exact versioned Gemma system, envelope, scalar, and item prompts;
- external JSON schemas with SHA-256 verification;
- task catalog containing only task IDs and model token limits;
- parallel scalar extraction, direct item extraction, read-only item contract diagnostics, and
  pure receipt assembly;
- inactive `StructuredExtractionStage` and standalone canonical-text runner.

## Production safety

The default production factory remains unchanged. Phase 3 does not replace `MainParsingStage`,
does not run deterministic receipt validation, and does not invoke any correction strategy. Its
purpose is to make the new extraction output executable and testable before Phase 4 integrates the
read-only validator.

## Preserved experiment behavior

- Gemma uses `/api/chat` with separate system and user messages.
- Every task uses the existing prompt, its existing schema, `temperature=0`, `seed=42`, and
  `think=false`.
- Scalar tasks use their original per-task `num_predict` values.
- Direct item extraction uses `num_predict=4096`.
- Model failures leave fields absent/null and are recorded; Python does not infer receipt values.
- Item contract checks are diagnostic only and never mutate the model result.
