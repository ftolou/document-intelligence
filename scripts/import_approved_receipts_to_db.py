#!/usr/bin/env python3
"""Import approved receipt artifacts into the receipt intelligence database.

Usage:
  python scripts/import_approved_receipts_to_db.py --results-dir var/jobs

This is useful for backfilling old jobs after adding the reviewed-data store. Human review imports
new approved receipts automatically through the web app.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import receipt_intelligence.settings as settings  # noqa: E402
from receipt_intelligence.storage.receipt_db import ReceiptDatabase  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Import approved receipt JSON files into SQLite.")
    parser.add_argument(
        "--results-dir", default=str(settings.RESULTS_DIR), help="Root output results directory."
    )
    parser.add_argument("--db", default=str(settings.RECEIPT_DB_PATH), help="SQLite DB path.")
    parser.add_argument(
        "--include-final",
        action="store_true",
        help="Also import final receipt JSON files when approved_receipt.json is missing.",
    )
    return parser.parse_args()


def candidate_files(results_dir: Path, include_final: bool) -> list[tuple[str, Path]]:
    candidates: list[tuple[str, Path]] = []
    for job_dir in sorted(p for p in results_dir.iterdir() if p.is_dir()):
        approved = job_dir / "approved_receipt.json"
        if approved.exists():
            candidates.append((job_dir.name, approved))
            continue
        if include_final:
            finals = sorted(
                job_dir.glob("*receipt_final*.json"), key=lambda p: p.stat().st_mtime, reverse=True
            )
            if finals:
                candidates.append((job_dir.name, finals[0]))
    return candidates


def main() -> int:
    args = parse_args()
    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f"Results dir does not exist: {results_dir}")
        return 2
    db = ReceiptDatabase(Path(args.db))
    imported = []
    failed = []
    for job_id, path in candidate_files(results_dir, args.include_final):
        try:
            receipt = json.loads(path.read_text(encoding="utf-8"))
            result = db.import_receipt(
                job_id=job_id, receipt=receipt, approved_receipt_path=path, source_receipt_path=path
            )
            imported.append({"job_id": job_id, "path": str(path), "item_count": result.item_count})
        except Exception as exc:
            failed.append({"job_id": job_id, "path": str(path), "error": str(exc)})
    print(
        json.dumps(
            {"db": str(args.db), "imported": imported, "failed": failed, "summary": db.summary()},
            ensure_ascii=False,
            indent=2,
        )
    )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
