#!/usr/bin/env python3
"""Run image folders through PaddleOCR and the receipt extraction pipeline."""

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
from receipt_intelligence.engines.ocr_engine import (  # noqa: E402
    IMAGE_EXTENSIONS,
    run_paddleocr_image,
)
from receipt_intelligence.pipeline.integrated_receipt_pipeline import (  # noqa: E402
    run_integrated_receipt_pipeline,
)


def progress(event: dict[str, Any]) -> None:
    stage = event.get("stage", "pipeline")
    status = event.get("status", "info")
    message = event.get("message", "")
    print(f"[{stage}] {status}: {message}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the receipt extraction pipeline on every receipt image in a folder"
    )
    parser.add_argument("image_folder", type=Path)
    parser.add_argument("--out-dir", type=Path, default=Path("var/reports/receipt_images_batch"))
    parser.add_argument("--ollama-url", default="http://localhost:11434")
    parser.add_argument("--model", default="gemma4:latest")
    parser.add_argument("--ocr-lang", default="german")
    parser.add_argument("--ocr-device", default="cpu")
    parser.add_argument("--ocr-max-side-limit", type=int, default=4000)
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
    parser.add_argument("--max-lines-for-llm", type=int, default=260)
    parser.add_argument("--vlm-backend", default="paddleocr_vl")
    parser.add_argument("--vlm-service-url", default="http://receipt-vlm:7870")
    parser.add_argument("--vlm-timeout-seconds", type=float, default=240.0)
    parser.add_argument("--disable-vlm-correction", action="store_true")
    parser.add_argument("--tolerance", type=float, default=0.03)
    args = parser.parse_args()

    images = sorted(p for p in args.image_folder.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
    if not images:
        raise SystemExit(f"No receipt images found in {args.image_folder}")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []

    for image_path in images:
        run_id = f"receipt_{image_path.stem}_{uuid.uuid4().hex[:6]}"
        run_dir = args.out_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n=== {image_path.name} -> {run_id} ===")
        try:
            ocr_json_path = run_dir / f"{run_id}_ocr_full_image.json"
            ocr = run_paddleocr_image(
                image_path=image_path,
                out_json_path=ocr_json_path,
                work_dir=run_dir,
                lang=args.ocr_lang,
                device=args.ocr_device,
                max_side_limit=args.ocr_max_side_limit,
                progress_callback=progress,
            )
            result = run_integrated_receipt_pipeline(
                ocr_json_path=ocr_json_path,
                result_dir=run_dir,
                run_id=run_id,
                ollama_url=args.ollama_url,
                model=args.model,
                tolerance=args.tolerance,
                ocr_lang=args.ocr_lang,
                ocr_device=args.ocr_device,
                max_lines_for_llm=args.max_lines_for_llm,
                num_ctx=args.num_ctx,
                num_predict=args.num_predict,
                keep_alive=args.keep_alive,
                llm_timeout_seconds=args.llm_timeout_seconds,
                json_retry_count=args.json_retry_count,
                format_json=not args.no_format_json,
                source_image_path=image_path,
                vlm_backend=args.vlm_backend,
                vlm_service_url=args.vlm_service_url,
                vlm_timeout_seconds=args.vlm_timeout_seconds,
                correction_enabled=not args.disable_vlm_correction,
                progress_callback=progress,
            )
            report = result["report"]
            rows.append(
                {
                    "file": image_path.name,
                    "run_id": run_id,
                    "state": "done",
                    "line_count": ocr.get("line_count"),
                    "balanced": report.get("balanced"),
                    "import_decision": report.get("import_decision"),
                    "difference": report.get("difference"),
                    "issue_count": len(report.get("issues") or []),
                    "correction_used": (result.get("pipeline_meta") or {})
                    .get("correction", {})
                    .get("used"),
                    "vlm_status": (result.get("pipeline_meta") or {}).get("vlm", {}).get("status"),
                    "primary_failure_mode": (report.get("failure_diagnosis") or {}).get(
                        "primary_failure_mode"
                    ),
                    "receipt_path": str(result["paths"].get("receipt_final_reconciled")),
                    "report_path": str(result["paths"].get("reconciliation_report")),
                }
            )
        except Exception as exc:
            rows.append(
                {"file": image_path.name, "run_id": run_id, "state": "error", "error": str(exc)}
            )
            print(f"ERROR: {exc}")

    summary_json = args.out_dir / "image_batch_summary.json"
    summary_csv = args.out_dir / "image_batch_summary.csv"
    summary_json.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    fieldnames = sorted({k for row in rows for k in row.keys()})
    with summary_csv.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {summary_json}")
    print(f"Wrote {summary_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
