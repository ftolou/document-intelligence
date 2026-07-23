# Phase 4: staged extraction workflow

The historical extraction entry point remains available:

```python
from receipt_intelligence.pipeline.integrated_receipt_pipeline import (
    run_integrated_receipt_pipeline,
)
```

Internally, it now constructs an `ExtractionConfig`, an `ExtractionContext`, and
runs `ReceiptExtractionWorkflow` through five explicit stages:

1. `PreparationStage`
2. `VisualEvidenceStage`
3. `MainParsingStage`
4. `RepairAndCorrectionStage`
5. `FinalizationStage`

The extraction algorithms and `_v14` implementation modules are intentionally
retained. Phase 4 changes orchestration and ownership boundaries, not receipt
semantics. Existing artifact filenames and the function return contract remain
compatible.

## Why this structure

- Every stage has one lifecycle responsibility.
- Shared mutable values are explicit fields on `ExtractionContext`.
- Configuration is collected in one typed dataclass.
- Stage duration and failures are recorded in
  `<run_id>_extraction_stage_trace.json`.
- The workflow can be tested with lightweight fake stages without OCR, VLM, or
  Ollama.
- Future extraction changes can replace one stage without editing a single
  1,000-line orchestration function.

## Compatibility policy

Modules such as `receipt_validation_v14.py` and
`receipt_table_assembler_v14.py` are still the authoritative implementations.
They should be renamed only in a later controlled phase after regression
coverage is expanded. Do not delete them manually.
