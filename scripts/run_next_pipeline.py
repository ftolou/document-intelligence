#!/usr/bin/env python3
"""Run the opt-in Phase 7 workflow through the production application entry point."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from receipt_intelligence.extraction import ExtractionRequest
from receipt_intelligence.pipeline.integrated_receipt_pipeline import run_receipt_extraction


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("--ocr-json", type=Path, required=True)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--ollama-url", default="http://localhost:11434")
    parser.add_argument("--gemma-model", default="gemma4:latest")
    parser.add_argument("--disable-correction", action="store_true")
    parser.add_argument("--disable-categorization", action="store_true")
    args = parser.parse_args()

    result = run_receipt_extraction(
        ExtractionRequest(
            ocr_json_path=args.ocr_json,
            result_dir=args.result_dir,
            run_id=args.run_id,
            ollama_url=args.ollama_url,
            model=args.gemma_model,
            source_image_path=args.image,
            correction_enabled=not args.disable_correction,
            categorization_enabled=not args.disable_categorization,
        ),
        extraction_strategy="next",
    )
    print(
        json.dumps(
            {
                "status": result.get("report", {}).get("status"),
                "paths": {key: str(value) for key, value in result.get("paths", {}).items()},
                "pipeline_meta": result.get("pipeline_meta"),
            },
            ensure_ascii=False,
            indent=2,
            default=str,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
