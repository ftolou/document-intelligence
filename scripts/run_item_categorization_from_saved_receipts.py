#!/usr/bin/env python3
"""Run V14.14 item categorization from saved final receipt JSON files.

This does not rerun OCR, VLM, receipt parsing, correction, or validation.
It reads existing final receipt JSON files and writes categorized copies.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
import time
import traceback
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
from typing import Any

from receipt_intelligence.extraction.categorization.items import (
    categorize_receipt_items_llm,
    write_categorization_artifacts,
)


def _load_json(path: Path) -> dict[str, Any]:
    obj = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(obj, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return obj


def _find_receipt_file(job_dir: Path) -> Path | None:
    candidates = [
        job_dir / "latest_receipt_final_reconciled.json",
        job_dir / "receipt_final_reconciled.json",
        job_dir / "latest_receipt_final.json",
    ]
    candidates.extend(sorted(job_dir.glob("*_receipt_final_reconciled.json")))
    candidates.extend(sorted(job_dir.glob("*_receipt_final.json")))
    seen: set[Path] = set()
    for p in candidates:
        if p in seen:
            continue
        seen.add(p)
        if p.exists() and p.is_file():
            return p
    return None


def _iter_job_dirs(results_root: Path) -> list[Path]:
    if _find_receipt_file(results_root):
        return [results_root]
    return [p for p in sorted(results_root.iterdir()) if p.is_dir() and _find_receipt_file(p)]


def _copy_source_artifacts(receipt_path: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        shutil.copyfile(receipt_path, out_dir / receipt_path.name)
    except Exception:
        pass


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    jobs = _iter_job_dirs(args.results_root)
    args.out_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for idx, job_dir in enumerate(jobs, start=1):
        run_id = job_dir.name
        receipt_path = _find_receipt_file(job_dir)
        row: dict[str, Any] = {"run_id": run_id, "source_dir": str(job_dir), "status": "unknown"}
        print(f"[{idx}/{len(jobs)}] {run_id} -> categorize")
        try:
            if receipt_path is None:
                raise FileNotFoundError(f"No final receipt JSON found in {job_dir}")
            receipt = _load_json(receipt_path)
            out_dir = args.out_root / run_id
            _copy_source_artifacts(receipt_path, out_dir)
            result = categorize_receipt_items_llm(
                receipt,
                ollama_url=args.ollama_url,
                model=args.model,
                num_ctx=args.num_ctx,
                num_predict=args.num_predict,
                keep_alive=args.keep_alive or None,
                timeout=args.timeout,
                format_json=not args.no_format_json,
            )
            paths = write_categorization_artifacts(result, result_dir=out_dir, run_id=run_id)
            categorized = result.get("receipt") or {}
            cat_meta = (
                categorized.get("categorization")
                if isinstance(categorized.get("categorization"), dict)
                else {}
            )
            row.update(
                {
                    "status": result.get("status"),
                    "item_count": cat_meta.get("item_count"),
                    "categorized_count": cat_meta.get("categorized_count"),
                    "duration_seconds": result.get("duration_seconds"),
                    "error": result.get("error"),
                    "receipt_final_categorized": str(paths.get("receipt_final_categorized")),
                    "categorization_result": str(paths.get("categorization_result")),
                }
            )
        except Exception as exc:
            tb = traceback.format_exc()
            err_path = args.out_root / run_id / f"{run_id}_categorization_error.json"
            err_path.parent.mkdir(parents=True, exist_ok=True)
            err_path.write_text(
                json.dumps(
                    {"error": f"{type(exc).__name__}: {exc}", "traceback": tb},
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            row.update(
                {
                    "status": "error",
                    "error": f"{type(exc).__name__}: {exc}",
                    "error_file": str(err_path),
                }
            )
            print(f"  ERROR: {row['error']}")
        rows.append(row)
    summary_csv = args.out_root / "categorization_summary.csv"
    fields = [
        "run_id",
        "status",
        "item_count",
        "categorized_count",
        "duration_seconds",
        "error",
        "receipt_final_categorized",
        "categorization_result",
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
        "duration_seconds": round(time.perf_counter() - started, 2),
        "summary_csv": str(summary_csv),
        "rows": rows,
    }
    (args.out_root / "categorization_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run V14.14 LLM-first item categorization from saved final receipts only."
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        required=True,
        help="Folder containing job result dirs or a single job dir",
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        required=True,
        help="Where categorized receipt JSON files should be written",
    )
    parser.add_argument("--ollama-url", default="http://localhost:11434")
    parser.add_argument("--model", default="gemma4")
    parser.add_argument("--num-ctx", type=int, default=8192)
    parser.add_argument("--num-predict", type=int, default=4096)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--keep-alive", default="")
    parser.add_argument("--no-format-json", action="store_true")
    args = parser.parse_args()
    summary = run(args)
    print(
        json.dumps({k: v for k, v in summary.items() if k != "rows"}, ensure_ascii=False, indent=2)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
