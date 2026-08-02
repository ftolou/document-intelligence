#!/usr/bin/env python3
"""Run the inactive Gemma structured-extraction subsystem on canonical text."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from receipt_intelligence.extraction.contracts.extraction import StructuredExtractionRequest
from receipt_intelligence.extraction.contracts.transcription import TranscriptionResult
from receipt_intelligence.extraction.settings import DEFAULT_SCALAR_TASKS, ParsingSettings
from receipt_intelligence.extraction.structured.composition import (
    build_gemma_structured_extraction_service,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("transcription", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("var/next_structured_extraction"))
    parser.add_argument("--run-id", default="single")
    parser.add_argument("--ollama-url", default="http://localhost:11434")
    parser.add_argument("--gemma-model", default="gemma4")
    parser.add_argument("--num-ctx", type=int, default=16384)
    parser.add_argument("--item-num-predict", type=int, default=4096)
    parser.add_argument("--parallelism", type=int, default=2)
    parser.add_argument("--scalar-tasks", nargs="+", default=list(DEFAULT_SCALAR_TASKS))
    parser.add_argument("--skip-scalars", action="store_true")
    parser.add_argument("--skip-items", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    text = args.transcription.read_text(encoding="utf-8").strip()
    transcription = TranscriptionResult(canonical_text=text, rows=(), crops=(), fragments=())
    settings = ParsingSettings(
        ollama_url=args.ollama_url,
        model=args.gemma_model,
        num_ctx=args.num_ctx,
        item_num_predict=args.item_num_predict,
        parallelism=args.parallelism,
        scalar_tasks=tuple(args.scalar_tasks),
        skip_scalars=args.skip_scalars,
        skip_items=args.skip_items,
    )
    service = build_gemma_structured_extraction_service(
        settings,
        result_dir=args.output_dir,
    )
    result = service.extract(
        StructuredExtractionRequest(run_id=args.run_id, transcription=transcription)
    )
    print(json.dumps(result.receipt, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
