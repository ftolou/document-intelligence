# Architecture

```mermaid
flowchart TD
    A[Receipt image] --> B[Paddle text detection]
    B --> C[Safe crop planning]
    C --> D[Qwen canonical transcription]
    D --> E[Gemma scalar and item extraction]
    E --> F[Read-only deterministic validation]
    F --> G[Source-evidence specialist correction]
    G --> H[Optional categorization]
    H --> I[Stable artifact publication]
    I --> J[Human review]
    J --> K[(Approved SQLite receipts)]

    Q[Natural-language question] --> L[RAG-SQL LangGraph]
    K --> M[Hybrid retrieval]
    M --> L
    K --> L
    L --> N[Validated read-only SQL]
    N --> O[Grounded answer]
```

## Extraction boundary

`run_receipt_extraction` accepts one immutable `ExtractionRequest` containing the source image
and model/runtime settings. `build_extraction_workflow` defines the only production workflow.
There is no runtime strategy selection, preliminary OCR job, remote VLM service, or fallback to
the removed OCR/VLM parser.

The workflow stages exchange typed artifacts through `ExtractionContext`. Model calls use
provider-neutral text and multimodal gateways, and Paddle is isolated behind the text-detection
port. Deterministic validation never mutates the receipt. Correction candidates are accepted only
when targeted validation failures improve without regressions.

## Application boundary

Flask blueprints parse transport data and call application use cases. Receipt and batch work is
submitted through the bounded `JobDispatcher`. Job state, attempts, errors, and resources are
persisted under `var/jobs`.

## Compatibility boundary

Stable final artifact names and receipt-schema adapters remain so existing jobs, review records,
and relational imports stay readable. Compatibility is limited to persisted data contracts; there
is no executable legacy extraction API or fallback pipeline.

## Query boundary

RAG-SQL resolves reviewed product concepts, produces typed plans, validates read-only SQL against
curated analytics views, and executes only approved statements. The LLM never executes SQL.

## Runtime services

```text
receipt-app  Flask UI/API, Paddle geometry, extraction, review, SQLite, and RAG-SQL
Ollama       Qwen/Gemma generation and embedding models on the host
```
