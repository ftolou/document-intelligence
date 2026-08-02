# Extraction pipeline integration — Phase 1 foundation

## Scope

This phase is deliberately additive. It establishes the typed destination architecture for the
Qwen/Gemma receipt pipeline without selecting it in production.

Added boundaries:

- grouped immutable pipeline settings;
- canonical transcription, extraction, validation, correction, and final result contracts;
- provider-neutral multimodal and text-detection ports;
- semantic artifact-store port;
- service protocols for transcription, structured extraction, validation, and correction;
- versioned, SHA-256-verified prompt registry;
- focused unit tests.

## Explicit non-goals

This phase does **not**:

- change `build_default_extraction_workflow()`;
- modify current stages or stage order;
- modify any existing prompt or response schema;
- replace PaddleOCR-VL or the current parser;
- connect the specialist correction coordinator;
- change `JobProcessingService`;
- change receipt output files, database writes, review flow, or observability.

## Migration seam

The current application seam remains:

```text
JobProcessingService
    -> ExtractionRequest
    -> run_receipt_extraction
    -> ReceiptExtractionWorkflow
```

`PipelineSettings.from_extraction_config(...)` is the compatibility bridge for later phases. It
requires the Qwen transcription model explicitly because the current flat `ExtractionConfig`
contains only the Gemma/main-parser model.

## Next phase

Phase 2 should implement the transcription adapters and service behind the new contracts, run them
from isolated tests/tools, and leave the active production factory unchanged until parity is shown.
