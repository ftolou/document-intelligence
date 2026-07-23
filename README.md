# Document Intelligence Pipeline

Local-first OCR/VLM/LLM document intelligence pipeline, currently specialized for receipts.

The current module processes receipt images with **PaddleOCR**, optional **PaddleOCR-VL**, **Ollama/Gemma**, deterministic validation, item categorization, batch execution, side-by-side auditable human review with receipt image and item correction, regression reporting, and a **receipt intelligence database + hybrid RAG search**. It is positioned as a portfolio-ready AI automation / KI Manager case study: the app demonstrates how an AI result can be extracted, validated, reviewed by a human, stored as structured business data, and queried later with evidence.

## What this project proves

- **AI workflow implementation:** image upload -> explicit extraction stages -> OCR/VLM evidence -> LLM extraction -> validation -> categorized JSON.
- **Human-in-the-loop control:** reviewers can correct key fields and save an auditable approval record.
- **Receipt intelligence database:** approved receipts and item lines are stored in SQLite as the source of truth.
- **RAG-SQL LangGraph Engine:** semantic product resolution, validated read-only SQL, bounded repair, deterministic answer extraction, and evidence-bound LLM fallback are orchestrated as an explicit graph.
- **Quality management:** batch execution and regression-report generation make failures measurable.
- **Local/DSGVO-aware architecture:** Ollama runs locally on the host; Docker services isolate the app and VLM layer.
- **Release discipline:** Docker Compose, PowerShell helpers, GitHub Actions syntax/build checks, and reproducible artifacts.


## Staged receipt extraction

The compatibility entry point now delegates to a five-stage application workflow:

```text
prepare -> visual evidence -> main parsing -> repair/correction -> finalize
```

Each run persists an extraction stage trace and an extraction metrics artifact while retaining all established
receipt artifact names. See [`docs/PROJECT_STRUCTURE.md`](docs/PROJECT_STRUCTURE.md) for the active package boundaries.

## Ask Your Receipts

After saving a human review, the app automatically imports the approved receipt
and item lines into `var/database/receipt_intelligence.db`. **Ask Your Receipts**
uses one production query path:

```text
question
  -> LangGraph question analysis
  -> semantic retrieval and candidate resolution when required
  -> SQL generation
  -> deterministic SQL validation and bounded repair
  -> read-only SQLite execution
  -> deterministic answer extraction
  -> bounded LLM normalization only for evidence-rich ambiguity
  -> deterministic evidence validation and final rendering
```

The engine uses reviewed receipt data and curated analytics views only. Product
identity is resolved to database item IDs before SQL planning. Reviewed
`semantic_description` and `category_reason` fields support grounded product
descriptions, product-type answers, and explicit brand identification. Seller
metadata is never treated as a product brand. Clear results stay on the fast
path without another LLM call. Ambiguous reviewed evidence may enter a bounded
structured formatter, but every returned value and supporting item ID is
validated against the SQL rows before deterministic rendering. Missing or
invalid evidence returns `insufficient_info` instead of an invented answer.

Useful commands:

```powershell
python scripts/demo_rag_sql_query.py "Wie viel habe ich für Schuhe ausgegeben?"
python scripts/run_tests.py
python scripts/import_approved_receipts_to_db.py --results-dir var/jobs
```

See [`docs/RAG_SQL_ENGINE.md`](docs/RAG_SQL_ENGINE.md) and
[`docs/RAG_SQL_PRODUCT_SEMANTICS.md`](docs/RAG_SQL_PRODUCT_SEMANTICS.md).

## RAG hybrid semantic retrieval

After building the approved item embedding index, inspect fused dense/lexical matches with:

```powershell
docker compose -f docker-compose.yml -f docker-compose.dev.yml exec receipt-app `
  python /app/scripts/search_rag_items.py "Schuhe" --limit 10
```

Hybrid retrieval combines dense and lexical evidence before bounded candidate
resolution. RAG-SQL is the only application query engine and uses LangGraph for
explicit routing, validated read-only SQL, bounded repair, and evidence-bound answer
formatting without slowing clear queries. See `docs/RAG_SEMANTIC_RETRIEVAL.md` and
`docs/RAG_SQL_ENGINE.md`.

## Documentation

See [`docs/index.md`](docs/index.md) for the current documentation set. Historical patch notes are under [`docs/archive/`](docs/archive/README.md).

## Canonical runtime layout

All generated state is stored under `var/`. New jobs include a `manifest.json`
that catalogs their artifacts. Legacy runtime directories are no longer mounted
or searched. Current runtime ownership and extraction package boundaries are documented in
[`docs/PROJECT_STRUCTURE.md`](docs/PROJECT_STRUCTURE.md). Historical implementation
notes remain under [`docs/archive/`](docs/archive/README.md).

## Runtime services

```text
receipt-app  Flask UI/API, CPU PaddleOCR, LLM extraction, validation, review, batch runner, SQLite receipt DB, Ask Your Receipts
receipt-vlm  Optional GPU PaddleOCR-VL evidence service
Ollama       Runs on the Windows host, e.g. http://host.docker.internal:11434
```

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for the architecture diagram.

## Quick start on Windows

Prerequisites:

- Docker Desktop with NVIDIA GPU support if you use the VLM service.
- Ollama running on the host machine.
- A local model available in Ollama, for example `gemma4`.
- Optional persistent model cache folder outside the repo.

First-time build:

```powershell
.\scripts\docker\build-app-runtime.ps1
.\scripts\docker\build-app.ps1
.\scripts\docker\build-vlm-runtime.ps1
.\scripts\docker\build-vlm.ps1
```

Start normally:

```powershell
docker compose up -d --no-build
```

Or use the helper:

```powershell
.\start_windows.ps1
```

Open the UI:

```text
http://localhost:7860
```

Check the VLM service:

```powershell
Invoke-RestMethod http://localhost:7870/health | ConvertTo-Json -Depth 5
```

## Local development without rebuilds

For normal Python/HTML/CSS/JS changes, use bind mounts and restart only the changed service:

```powershell
.\scripts\docker\start.ps1
```

Restart app code:

```powershell
.\scripts\docker\restart-app.ps1
```

Restart VLM code:

```powershell
.\scripts\docker\restart-vlm.ps1
```

Restart both app and VLM:

```powershell
.\scripts\docker\restart-all.ps1
```

If PowerShell blocks script execution on your machine, run the command in the current session with:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

## Human review workflow

After a receipt job is finished, the UI shows a **Human review** section with the original receipt image on the left and editable extraction fields on the right. The reviewer can correct high-value receipt fields and item rows such as product name, category, quantity, unit price and line total. Saving the review creates these artifacts inside the job folder:

```text
approved_receipt.json
human_review_record.json
```

The review record contains reviewer, status, notes, timestamp, changed fields, submitted field values, submitted item corrections, and the database import result. For an already imported receipt, the editor reads and writes SQLite directly by `receipt_id`; job JSON is only a best-effort mirror. Product-name, normalized-name, parser-row-type, reviewed-category, category-reason, or semantic-description changes invalidate the affected vectors and selectively reindex those item IDs. Merchant, date, quantity, and monetary edits update relational/FTS data without re-embedding.

## Batch execution and regression report

Put test receipt images into:

```text
./var/batch_input
```

Run a batch from the UI with server folder:

```text
/app/var/batch_input
```

Generate a regression report from saved jobs:

```powershell
python .\scripts\generate_regression_report.py --results-dir .\var\jobs --out-dir .\var\reports\regression_report
```

The script writes:

```text
var/reports/regression_report/regression_summary.json
var/reports/regression_report/regression_jobs.csv
var/reports/regression_report/regression_report.md
```

This turns the project into a measurable AI-quality case study instead of a one-off OCR demo.

## Portfolio documentation

- [`docs/archive/portfolio/PORTFOLIO_PHASE1.md`](docs/archive/portfolio/PORTFOLIO_PHASE1.md) — what was added for Phase 1 and how to present it.
- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — architecture diagram and data flow.
- [`docs/HUMAN_REVIEW.md`](docs/HUMAN_REVIEW.md) — review API and audit artifacts.
- [`docs/REGRESSION_REPORTING.md`](docs/REGRESSION_REPORTING.md) — batch/regression reporting concept.
- [`docs/SCREENSHOT_GUIDE.md`](docs/SCREENSHOT_GUIDE.md) — screenshots to capture for applications.

## Repository structure

```text
├── src/receipt_intelligence/       Python package, including observability
├── static/                         Browser UI
├── docker/                         Runtime and thin app Dockerfiles
├── scripts/                        Build, restart, batch, and report helpers
├── docs/                           Portfolio and technical documentation
└── var/                            Runtime jobs, DB, reports, uploads, and logs
```

## Rebuild rules

Rebuild runtime images only when dependencies or base-image strategy changes:

```text
requirements/*.txt
docker/Dockerfile.app-runtime
docker/Dockerfile.vlm-runtime-cu126
CUDA / Paddle / PaddleOCR / PaddleX / Torch dependency strategy
system apt packages
```

For normal source-code changes, use development bind mounts and restart the service instead of rebuilding heavy images. Phase 7 changes app dependency constraints, so upgrading from v1.21 requires one app-runtime rebuild; the VLM image remains unchanged.


## Current extraction capabilities

The active extraction pipeline includes dedicated table interpretation, OCR/VLM row arbitration, validation-gated right-column and vertical price-stack recovery, compact patch-only correction, duplicate-aware review queues, and approved-receipt database import. Current operating documentation is indexed in [`docs/index.md`](docs/index.md); detailed release chronology is archived in [`docs/archive/RELEASE_HISTORY.md`](docs/archive/RELEASE_HISTORY.md).
