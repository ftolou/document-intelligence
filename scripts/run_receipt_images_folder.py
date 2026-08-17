#!/usr/bin/env python3
"""Run a folder of receipt images through the canonical extraction pipeline."""

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

from receipt_intelligence.extraction import ExtractionRequest  # noqa: E402
from receipt_intelligence.pipeline import run_receipt_extraction  # noqa: E402

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}


def progress(event: dict[str, Any]) -> None:
    print(
        f"[{event.get('stage', 'pipeline')}] "
        f"{event.get('status', 'info')}: {event.get('message', '')}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image_folder", type=Path)
    parser.add_argument("--out-dir", type=Path, default=Path("var/reports/receipt_images_batch"))
    parser.add_argument("--ollama-url", default="http://localhost:11434")
    parser.add_argument("--model", default="gemma4:latest")
    parser.add_argument("--ocr-lang", default="german")
    parser.add_argument("--ocr-device", default="cpu")
    parser.add_argument("--max-crops", type=int, default=4)
    parser.add_argument("--num-ctx", type=int, default=24384)
    parser.add_argument("--num-predict", type=int, default=8192)
    parser.add_argument("--keep-alive", default="10m")
    parser.add_argument("--llm-timeout-seconds", type=float, default=300.0)
    parser.add_argument("--disable-correction", action="store_true")
    parser.add_argument("--disable-categorization", action="store_true")
    args = parser.parse_args()

    images = sorted(
        path for path in args.image_folder.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not images:
        raise SystemExit(f"No receipt images found in {args.image_folder}")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []

    for image_path in images:
        run_id = f"receipt_{image_path.stem}_{uuid.uuid4().hex[:6]}"
        run_dir = args.out_dir / run_id
        try:
            result = run_receipt_extraction(
                ExtractionRequest(
                    source_image_path=image_path,
                    result_dir=run_dir,
                    run_id=run_id,
                    ollama_url=args.ollama_url,
                    model=args.model,
                    ocr_lang=args.ocr_lang,
                    ocr_device=args.ocr_device,
                    max_crops=args.max_crops,
                    num_ctx=args.num_ctx,
                    num_predict=args.num_predict,
                    keep_alive=args.keep_alive,
                    llm_timeout_seconds=args.llm_timeout_seconds,
                    correction_enabled=not args.disable_correction,
                    categorization_enabled=not args.disable_categorization,
                    progress_callback=progress,
                )
            )
            report = result.get("report", {})
            rows.append(
                {
                    "file": image_path.name,
                    "run_id": run_id,
                    "state": "done",
                    "status": report.get("status"),
                    "receipt_path": str(result["paths"].get("receipt_final")),
                    "report_path": str(result["paths"].get("validation_report")),
                }
            )
        except Exception as exc:
            rows.append(
                {"file": image_path.name, "run_id": run_id, "state": "error", "error": str(exc)}
            )

    summary_json = args.out_dir / "image_batch_summary.json"
    summary_csv = args.out_dir / "image_batch_summary.csv"
    summary_json.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    fieldnames = sorted({key for row in rows for key in row})
    with summary_csv.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {summary_json}")
    print(f"Wrote {summary_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
