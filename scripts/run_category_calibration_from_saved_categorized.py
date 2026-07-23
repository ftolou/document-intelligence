#!/usr/bin/env python3
"""Apply V14.14.1 category confidence calibration to saved categorized receipts.

This does not call Ollama and does not rerun OCR/VLM/parser/categorization.
It only backfills/updates:
- category_confidence_raw
- category_confidence_calibrated
- category_review_required
- category_review_reasons
- categorization.category_review_count
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
from typing import Any

from receipt_intelligence.extraction.categorization.items import (
    recalibrate_existing_categorized_receipt,
)


def _load_json(path: Path) -> dict[str, Any]:
    obj = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(obj, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return obj


def _save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _find_categorized_file(job_dir: Path) -> Path | None:
    candidates = [
        job_dir / "latest_receipt_final_categorized.json",
        job_dir / "receipt_final_categorized.json",
    ]
    candidates.extend(sorted(job_dir.glob("*_receipt_final_categorized.json")))
    seen: set[Path] = set()
    for p in candidates:
        if p in seen:
            continue
        seen.add(p)
        if p.exists() and p.is_file():
            return p
    return None


def _iter_job_dirs(root: Path) -> list[Path]:
    if _find_categorized_file(root):
        return [root]
    return [p for p in sorted(root.iterdir()) if p.is_dir() and _find_categorized_file(p)]


def run(args: argparse.Namespace) -> dict[str, Any]:
    jobs = _iter_job_dirs(args.results_root)
    args.out_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for idx, job_dir in enumerate(jobs, start=1):
        run_id = job_dir.name
        print(f"[{idx}/{len(jobs)}] {run_id} -> recalibrate categories")
        src = _find_categorized_file(job_dir)
        row: dict[str, Any] = {"run_id": run_id, "source_dir": str(job_dir), "status": "unknown"}
        try:
            if src is None:
                raise FileNotFoundError(f"No categorized receipt JSON found in {job_dir}")
            receipt = _load_json(src)
            calibrated = recalibrate_existing_categorized_receipt(receipt)
            out_dir = args.out_root / run_id
            out_dir.mkdir(parents=True, exist_ok=True)
            try:
                shutil.copyfile(src, out_dir / src.name)
            except Exception:
                pass
            out_path = out_dir / "latest_receipt_final_categorized.json"
            _save_json(out_path, calibrated)
            _save_json(out_dir / f"{run_id}_receipt_final_categorized.json", calibrated)
            items = calibrated.get("items") if isinstance(calibrated.get("items"), list) else []
            review_count = sum(
                1
                for item in items
                if isinstance(item, dict) and item.get("category_review_required")
            )
            row.update(
                {
                    "status": "ok",
                    "item_count": len(items),
                    "category_review_count": review_count,
                    "output": str(out_path),
                }
            )
        except Exception as exc:
            row.update({"status": "error", "error": f"{type(exc).__name__}: {exc}"})
            print(f"  ERROR: {row['error']}")
        rows.append(row)
    summary_csv = args.out_root / "category_calibration_summary.csv"
    fields = [
        "run_id",
        "status",
        "item_count",
        "category_review_count",
        "error",
        "output",
        "source_dir",
    ]
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    summary = {
        "results_root": str(args.results_root),
        "out_root": str(args.out_root),
        "job_count": len(rows),
        "status_counts": {
            status: sum(1 for row in rows if row.get("status") == status)
            for status in sorted({str(row.get("status")) for row in rows})
        },
        "summary_csv": str(summary_csv),
        "rows": rows,
    }
    _save_json(args.out_root / "category_calibration_summary.json", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Backfill V14.14.1 category confidence calibration on saved categorized receipts."
    )
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    args = parser.parse_args()
    summary = run(args)
    print(
        json.dumps({k: v for k, v in summary.items() if k != "rows"}, ensure_ascii=False, indent=2)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
