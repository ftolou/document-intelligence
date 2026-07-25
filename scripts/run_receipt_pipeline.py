#!/usr/bin/env python3
"""CLI wrapper for the V14 receipt pipeline."""

from __future__ import annotations

import argparse
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


def progress(event: dict[str, Any]) -> None:
    stage = event.get("stage", "pipeline")
    status = event.get("status", "info")
    msg = event.get("message", "")
    details = event.get("details") or {}
    tail = ""
    if details:
        small = {
            k: v
            for k, v in details.items()
            if k
            in {
                "balanced",
                "import_decision",
                "difference",
                "issue_count",
                "parse_status",
                "item_count",
                "duration_seconds",
            }
        }
        if small:
            tail = " " + json.dumps(small, ensure_ascii=False)
    print(f"[{stage}] {status}: {msg}{tail}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run V14 pure LLM-main receipt parser + deterministic validation"
    )
    parser.add_argument("ocr_json", type=Path, help="Full-image OCR JSON from the app/PaddleOCR")
    parser.add_argument("--out-dir", type=Path, default=Path("var/reports/manual_pipeline"))
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--ollama-url", default="http://localhost:11434")
    parser.add_argument("--model", default="gemma4")
    parser.add_argument("--tolerance", type=float, default=0.03)
    parser.add_argument("--max-lines-for-llm", type=int, default=260)
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
    parser.add_argument(
        "--source-image",
        type=Path,
        default=None,
        help="Original receipt image path for optional VLM evidence",
    )
    parser.add_argument(
        "--enable-vlm",
        action="store_true",
        help="Run optional PaddleOCR-VL/VLM evidence after validation failures",
    )
    parser.add_argument("--vlm-backend", default="paddleocr_vl")
    parser.add_argument("--vlm-service-url", default="http://receipt-vlm:7870")
    parser.add_argument(
        "--vlm-command", default="", help="Optional command template with {image} and {output_json}"
    )
    parser.add_argument("--vlm-timeout-seconds", type=float, default=240.0)
    parser.add_argument("--disable-vlm-correction", action="store_true")
    args = parser.parse_args()

    run_id = args.run_id or uuid.uuid4().hex[:12]
    result = run_integrated_receipt_pipeline(
        ocr_json_path=args.ocr_json,
        result_dir=args.out_dir,
        run_id=run_id,
        ollama_url=args.ollama_url,
        model=args.model,
        tolerance=args.tolerance,
        max_lines_for_llm=args.max_lines_for_llm,
        num_ctx=args.num_ctx,
        num_predict=args.num_predict,
        keep_alive=args.keep_alive,
        llm_timeout_seconds=args.llm_timeout_seconds,
        json_retry_count=args.json_retry_count,
        format_json=not args.no_format_json,
        source_image_path=args.source_image,
        vlm_enabled=args.enable_vlm,
        vlm_backend=args.vlm_backend,
        vlm_service_url=args.vlm_service_url,
        vlm_command=args.vlm_command,
        vlm_timeout_seconds=args.vlm_timeout_seconds,
        correction_enabled=not args.disable_vlm_correction,
        progress_callback=progress,
    )
    report = result["report"]
    paths = result["paths"]
    print("\nSummary")
    print(
        json.dumps(
            {
                "run_id": run_id,
                "balanced": report.get("balanced"),
                "import_decision": report.get("import_decision"),
                "difference": report.get("difference"),
                "issue_count": len(report.get("issues") or []),
                "failure_diagnosis": report.get("failure_diagnosis"),
                "receipt_final_reconciled": str(paths.get("receipt_final_reconciled")),
                "validation_report": str(paths.get("validation_report")),
                "llm_prompt": str(paths.get("llm_main_prompt")),
                "llm_raw": str(paths.get("llm_main_raw")),
                "visual_evidence": str(paths.get("visual_evidence")),
                "correction_patch_raw": str(paths.get("correction_patch_raw")),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if report.get("import_decision") in {"import", "needs_review"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
