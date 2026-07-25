#!/usr/bin/env python3
"""Run the receipt pipeline over a folder of saved OCR JSON files.

This runner does not execute PaddleOCR. It is intended for repeatable regression
runs against previously generated OCR inputs.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import uuid
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
from receipt_intelligence.pipeline.integrated_receipt_pipeline import (  # noqa: E402
    run_integrated_receipt_pipeline,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the receipt pipeline on a folder of OCR JSON files"
    )
    parser.add_argument("ocr_json_folder", type=Path)
    parser.add_argument("--out-dir", type=Path, default=Path("var/reports/receipt_batch"))
    parser.add_argument("--ollama-url", default="http://localhost:11434")
    parser.add_argument("--model", default="gemma4")
    parser.add_argument("--tolerance", type=float, default=0.03)
    parser.add_argument("--num-ctx", type=int, default=24384)
    parser.add_argument("--num-predict", type=int, default=8192)
    parser.add_argument("--keep-alive", default="10m")
    parser.add_argument("--llm-timeout-seconds", type=float, default=300.0)
    parser.add_argument("--json-retry-count", type=int, default=1)
    parser.add_argument(
        "--no-format-json",
        action="store_true",
        help="Disable Ollama JSON grammar; prompt still asks for JSON",
    )
    args = parser.parse_args()

    files = sorted(args.ocr_json_folder.glob("*.json"))
    if not files:
        raise SystemExit(f"No *.json files found in {args.ocr_json_folder}")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []

    for path in files:
        run_id = f"receipt_{path.stem}_{uuid.uuid4().hex[:6]}"
        print(f"\n=== {path.name} -> {run_id} ===")
        result = run_integrated_receipt_pipeline(
            ocr_json_path=path,
            result_dir=args.out_dir,
            run_id=run_id,
            ollama_url=args.ollama_url,
            model=args.model,
            tolerance=args.tolerance,
            num_ctx=args.num_ctx,
            num_predict=args.num_predict,
            keep_alive=args.keep_alive,
            llm_timeout_seconds=args.llm_timeout_seconds,
        )
        report = result["report"]
        rows.append(
            {
                "file": path.name,
                "run_id": run_id,
                "balanced": report.get("balanced"),
                "import_decision": report.get("import_decision"),
                "difference": report.get("difference"),
                "item_count": report.get("item_count"),
                "priced_item_count": report.get("priced_item_count"),
                "issue_count": len(report.get("issues") or []),
                "primary_failure_mode": (report.get("failure_diagnosis") or {}).get(
                    "primary_failure_mode"
                ),
                "receipt_path": str(result["paths"].get("receipt_final_reconciled")),
                "report_path": str(result["paths"].get("reconciliation_report")),
            }
        )

    summary_json = args.out_dir / "batch_summary.json"
    summary_csv = args.out_dir / "batch_summary.csv"
    summary_json.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    with summary_csv.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {summary_json}")
    print(f"Wrote {summary_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
