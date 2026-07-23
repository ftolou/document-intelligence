# Phase 6: legacy cleanup and canonical runtime cutover

Phase 6 completes the migration begun in Phase 5. The application now uses one
runtime root only:

```text
var/
├── uploads/
├── jobs/
├── database/
├── reports/
├── batch_input/
└── logs/
```

## Removed compatibility layers

- Runtime reads from `outputs/results`, `data`, `uploads`, and `batch_input`
- Automatic copying of the old SQLite database at application startup
- Legacy runtime mounts in Docker Compose
- The `receipt_intelligence.query` import alias package
- Superseded root-level Docker helper wrappers
- Phase-specific documentation from the active documentation root

The query engine's internal `query/legacy` planner and executor are intentionally
retained. They still provide the configured one-shot fallback and are unrelated
to the removed `receipt_intelligence.query` package.

The active `_v14` extraction modules are also retained. Their names are historic,
but they still contain production extraction algorithms.

## Upgrade from v1.20.0

Stop both services and extract the Phase 6 patch over the project. Then run:

```powershell
.\scripts\apply_phase6_cleanup.ps1
```

The script performs an idempotent copy into `var/`, writes
`var/migration/phase6_cutover_report.json`, creates missing job manifests,
archives the old runtime folders next to the repository, and removes obsolete
compatibility files. It refuses to continue when destination conflicts are
found unless `-OverwriteConflicts` is explicitly supplied.

Recreate both containers after cleanup because the Compose mount list changed:

```powershell
docker compose -f docker-compose.yml -f docker-compose.dev.yml up -d `
  --no-build --force-recreate receipt-app receipt-vlm
```

No image rebuild is required.

## Validation

```powershell
python scripts/verify_runtime_layout.py
python scripts/run_quality_checks.py

docker compose -f docker-compose.yml -f docker-compose.dev.yml exec receipt-app `
  python /app/scripts/run_tests.py
```

The runtime verification output must show only `var/jobs` as the job read root.
