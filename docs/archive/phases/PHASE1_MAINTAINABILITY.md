# Phase 1 Maintainability Cleanup

This release performs structural cleanup without changing receipt extraction or query semantics.

## Changes

- Tests live under `tests/` and can run with either pytest or the standard-library runner.
- `receipt_intelligence.query` is the canonical query-engine package.
- `receipt_intelligence.qa` remains a compatibility alias for existing integrations.
- Docker helper scripts are separated by app and VLM deployment unit under `scripts/docker/`.
- Ruff, mypy, pytest, and local pre-commit configuration are defined centrally.
- Historical release notes were moved under `docs/archive/`.

## Commands

```powershell
python scripts/run_tests.py
python -m pytest
python scripts/run_quality_checks.py
```

Build only the app dependency image:

```powershell
.\scripts\docker\build-app-runtime.ps1
```

Build only the heavy VLM dependency image:

```powershell
.\scripts\docker\build-vlm-runtime.ps1
```

## Compatibility window

Imports from `receipt_intelligence.qa` are supported during the migration period. New code and tests must use `receipt_intelligence.query`. The compatibility alias can be removed in a later major cleanup after downstream callers have migrated.
