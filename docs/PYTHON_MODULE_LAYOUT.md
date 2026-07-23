# Python module layout

The project uses a `src/` package layout.

```text
src/receipt_intelligence/
├── app.py                         package application entry point
├── app_version.py                 VERSION file reader
├── settings.py                    runtime configuration
├── extraction/
│   ├── evidence/                  compact/grouped/layout/visual evidence
│   ├── parsing/                   LLM parser and table processing
│   ├── validation/                receipt validation and consistency processing
│   ├── repair/                    patch correction and bounded re-OCR recovery
│   ├── categorization/            receipt item categorization
│   └── stages/                    staged workflow nodes
├── pipeline/
│   └── integrated_receipt_pipeline.py  stable public extraction entry point
├── rag_sql/                       RAG-SQL LangGraph query implementation
├── runtime/                       canonical var paths and manifests
├── services/                      application services
├── storage/                       SQLite repositories and migrations
├── web/                           Flask factory and blueprints
└── utils/
```

Use the stable extraction entry point:

```python
from receipt_intelligence.pipeline.integrated_receipt_pipeline import (
    run_integrated_receipt_pipeline,
)
```

New extraction code should import implementation components from the relevant
`receipt_intelligence.extraction.*` package. Versioned module paths under
`receipt_intelligence.pipeline`, such as `receipt_validation_v14`, no longer
exist.

Use the application query service for HTTP or embedded execution:

```python
from receipt_intelligence.rag_sql.application import ReceiptQueryService
```
