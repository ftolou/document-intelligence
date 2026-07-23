# Phase 5: runtime paths, manifests, and legacy migration

Phase 5 separates generated runtime data from source code. New application data
is written below one root:

```text
var/
├── uploads/
├── jobs/
├── database/
├── reports/
├── batch_input/
└── logs/
```

The application defaults to `RUNTIME_LAYOUT=var`. Both Docker services mount
`./var` at `/app/var`, so the main app and VLM service see the same job images
and artifacts.

## Safe migration

Stop the app before copying the SQLite database and jobs:

```powershell
docker compose -f docker-compose.yml -f docker-compose.dev.yml stop receipt-app
.\scripts\migrate_runtime_layout.ps1
.\scripts\verify_runtime_layout.ps1
```

The migration is idempotent and copy-only by default. It preserves `uploads/`,
`outputs/`, `data/`, and `batch_input/`; conflicting destination files are
reported and not overwritten. Use `-Overwrite` only after reviewing the report.

Then restart the app without rebuilding images:

```powershell
docker compose -f docker-compose.yml -f docker-compose.dev.yml up -d --no-build --no-deps receipt-app
```

## Compatibility window

Historical jobs remain visible because `JobStore` reads from both:

```text
var/jobs
outputs/results
```

New jobs are always written to `var/jobs`. Existing legacy jobs may still be
reviewed in their original directory. The old mounts should be removed only
after migration and verification on the user's real dataset.

If `var/database/receipt_intelligence.db` is absent but the old database exists,
the app copies it once from `data/receipt_intelligence.db`. Set
`RUNTIME_AUTO_COPY_LEGACY_DB=0` to disable this safety behavior.

Rollback remains possible by setting:

```env
RUNTIME_LAYOUT=legacy
```

## Job manifests

Every new or accessed job can contain:

```text
manifest.json
```

The manifest uses `job_manifest_v1` and records:

- job identity and type,
- current state,
- whether it is still in the legacy layout,
- relative artifact paths,
- artifact category, MIME type, size, and update time,
- lightweight job metadata.

The manifest catalogs the existing flat artifact filenames; Phase 5 deliberately
does not move individual extraction artifacts into subdirectories because doing
so would change established pipeline and UI contracts.

Inspect a manifest through:

```text
GET /api/jobs/<job_id>/manifest
```

## Runtime configuration

Important variables:

```env
RUNTIME_LAYOUT=var
VAR_DIR=/app/var
RUNTIME_LEGACY_READ_ENABLED=1
RUNTIME_AUTO_COPY_LEGACY_DB=1
UPLOAD_DIR=/app/var/uploads
RESULTS_DIR=/app/var/jobs
DATA_DIR=/app/var/database
RECEIPT_DB_PATH=/app/var/database/receipt_intelligence.db
REPORTS_DIR=/app/var/reports
BATCH_INPUT_DIR=/app/var/batch_input
LOGS_DIR=/app/var/logs
```

`GET /api/config` exposes the resolved runtime layout and paths for diagnostics.
