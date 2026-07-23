# Development

## Start and restart

```powershell
.\scripts\docker\start.ps1
.\scripts\docker\restart-app.ps1
.\scripts\docker\restart-vlm.ps1
```

The legacy helper names in `scripts/` remain as compatibility wrappers.

## Install development tools

```powershell
python -m pip install -r requirements/app.txt -r requirements/dev.txt
```

## Test and lint

```powershell
python scripts/run_tests.py
python -m pytest
python scripts/run_quality_checks.py
python -m mypy src/receipt_intelligence/rag_sql
```

Inside the app container, the standard-library fallback does not require pytest:

```powershell
docker compose -f docker-compose.yml -f docker-compose.dev.yml exec receipt-app `
  python scripts/run_tests.py
```

## Build only what changed

| Change | Command |
|---|---|
| `requirements/app.txt` or app runtime Dockerfile | `.\scripts\docker\build-app-runtime.ps1`, then `.\scripts\docker\build-app.ps1` |
| App source/static only | restart app in dev mode, or `.\scripts\docker\build-app.ps1` |
| `requirements/vlm-gpu-cu126.txt` or VLM runtime Dockerfile | `.\scripts\docker\build-vlm-runtime.ps1`, then `.\scripts\docker\build-vlm.ps1` |
| VLM source only | restart VLM in dev mode, or `.\scripts\docker\build-vlm.ps1` |
| Both complete image families | `.\scripts\docker\build-all.ps1` |

Use `.\scripts\docker\rebuild-app.ps1` or `rebuild-vlm.ps1` for a one-command dependency rebuild, thin-image build, and service recreation.


## Test profiles

```powershell
python scripts/run_test_profile.py unit
python scripts/run_test_profile.py integration
python scripts/run_test_profile.py regression
python scripts/run_test_profile.py fast
```

After an app-runtime dependency rebuild, validate Requests compatibility with:

```powershell
python scripts/check_dependency_compatibility.py
```
