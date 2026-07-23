#!/usr/bin/env python3
"""Verify the canonical var-only runtime layout."""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from receipt_intelligence.runtime.paths import RuntimePaths  # noqa: E402

LEGACY_ROOTS = ("uploads", "outputs", "data", "batch_input")


def _count_job_dirs(root: Path) -> tuple[int, int]:
    if not root.exists():
        return 0, 0
    jobs = [path for path in root.iterdir() if path.is_dir()]
    manifests = sum(1 for path in jobs if (path / "manifest.json").exists())
    return len(jobs), manifests


def main() -> int:
    paths = RuntimePaths.from_environment(PROJECT_ROOT)
    paths.ensure_directories()
    job_count, manifest_count = _count_job_dirs(paths.jobs_dir)
    remaining_legacy_roots = [name for name in LEGACY_ROOTS if (PROJECT_ROOT / name).exists()]
    payload = {
        "runtime": paths.as_dict(),
        "jobs": {
            "path": str(paths.jobs_dir),
            "job_count": job_count,
            "manifest_count": manifest_count,
        },
        "receipt_database_exists": paths.receipt_db_path.exists(),
        "receipt_database_size_bytes": (
            paths.receipt_db_path.stat().st_size if paths.receipt_db_path.exists() else 0
        ),
        "remaining_legacy_roots": remaining_legacy_roots,
        "ok": not remaining_legacy_roots,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if payload["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
