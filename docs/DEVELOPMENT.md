# Development

## Start and restart

```powershell
.\scripts\docker\start.ps1
.\scripts\docker\restart-app.ps1
```

## Install development tools

```powershell
python -m pip install -r requirements/app.txt -r requirements/dev.txt
```

Test and lint tooling is intentionally kept in `requirements/dev.txt` rather than the production
application image.

## Test and lint

```powershell
python scripts/run_tests.py
python scripts/run_test_profile.py fast
python scripts/run_quality_checks.py
python -m mypy src/receipt_intelligence/rag_sql
```

`run_tests.py` requires pytest and fails with an installation hint when development dependencies are
missing; it does not fall back to a partial unittest run.

To validate the same Python 3.11 runtime used by the application image:

```powershell
docker compose `
  -f docker-compose.yml `
  -f docker-compose.dev.yml `
  run --rm receipt-app `
  sh -lc "python -m pip install -r requirements/dev.txt && python scripts/run_quality_checks.py && python scripts/run_tests.py"
```

## Build only what changed

| Change | Command |
|---|---|
| Application dependencies or runtime Dockerfile | `.\scripts\docker\build-app-runtime.ps1`, then `.\scripts\docker\build-app.ps1` |
| Python/static source only | `.\scripts\docker\build-app.ps1` or restart the development container |

The application targets Python 3.11 and Ruff `py311`.
