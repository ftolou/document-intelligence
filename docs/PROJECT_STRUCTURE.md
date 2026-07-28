# Project structure

```text
document-intelligence-pipeline/
├── src/receipt_intelligence/
│   ├── extraction/
│   │   ├── evidence/       OCR, layout, and VLM evidence preparation
│   │   ├── parsing/        LLM parser and table interpretation/assembly
│   │   ├── validation/     deterministic validation and consistency checks
│   │   ├── repair/         bounded re-OCR and validation-gated repair
│   │   ├── categorization/ item categorization after extraction
│   │   └── stages/         explicit extraction workflow stages
│   ├── observability/      compatibility serializers and timing helpers
│   ├── pipeline/           stable integrated pipeline entry point only
│   ├── rag_sql/            Query contracts, semantic resolution, safe SQL, formatting, and isolated orchestration adapters
│   ├── runtime/            canonical paths, readiness, and job manifests
│   ├── application/        ports, neutral events, resource contracts, and use cases
│   ├── adapters/           model, storage, jobs, lifecycle, and event-sink adapters
│   ├── services/           reusable application workflow implementations
│   ├── storage/            SQLite repositories and migrations
│   └── web/                Flask transport, presenters, and blueprint adapters
├── tests/                  unit, integration, and regression tests
├── scripts/
│   └── docker/             service-specific Docker operations
├── docker/                 runtime and thin-image Dockerfiles
├── static/                 browser UI
├── docs/                   current documentation
└── var/                    all generated runtime state
```

The only supported generated-data root is `var/`. Old roots such as `outputs/`,
`data/`, `uploads/`, and `batch_input/` are not mounted or searched.

## Extraction package boundaries

Active extraction code is organized by responsibility rather than release number:

- `extraction/evidence/` prepares OCR, layout, and mandatory PaddleOCR-VL evidence;
- `extraction/parsing/` owns LLM receipt parsing and table interpretation;
- `extraction/validation/` applies deterministic consistency checks;
- `extraction/repair/` contains bounded, validation-gated re-OCR and correction;
- `extraction/categorization/` categorizes reviewed item data;
- `extraction/stages/` orchestrates the extraction workflow.

The `pipeline/` package contains only the stable compatibility entry point
`integrated_receipt_pipeline.py`. Active algorithms do not use release-numbered
module names.

## Compatibility contracts

Receipt schema identifiers and historical artifact filenames may still contain
`v14`. They are persisted data contracts used by saved jobs, the review UI, and
regression fixtures; they are independent of the application version and Python
module names.

The supported correction path is the constrained patch-correction component under
`extraction/repair/patch_correction.py`. Active batch entry points are
`scripts/run_receipt_folder.py` and `scripts/run_receipt_images_folder.py`.
