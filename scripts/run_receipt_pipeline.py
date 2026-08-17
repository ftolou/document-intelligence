#!/usr/bin/env python3
"""Run the canonical image-first receipt extraction pipeline."""

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

from receipt_intelligence.extraction import ExtractionRequest  # noqa: E402
from receipt_intelligence.pipeline import run_receipt_extraction  # noqa: E402


def progress(event: dict[str, Any]) -> None:
    print(
        f"[{event.get('stage', 'pipeline')}] "
        f"{event.get('status', 'info')}: {event.get('message', '')}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("--out-dir", type=Path, default=Path("var/reports/manual_pipeline"))
    parser.add_argument("--run-id")
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

    run_id = args.run_id or uuid.uuid4().hex[:12]
    result = run_receipt_extraction(
        ExtractionRequest(
            source_image_path=args.image,
            result_dir=args.out_dir,
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
    print(
        json.dumps(
            {
                "run_id": run_id,
                "status": result.get("report", {}).get("status"),
                "paths": {key: str(value) for key, value in result.get("paths", {}).items()},
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
