#!/usr/bin/env python3
"""Run only the inactive next-pipeline Paddle/Qwen transcription subsystem."""

from __future__ import annotations

import argparse
from pathlib import Path

from receipt_intelligence.extraction.contracts.transcription import TranscriptionRequest
from receipt_intelligence.extraction.settings import (
    CategorizationSettings,
    CorrectionSettings,
    CropPlanningSettings,
    DetectionSettings,
    ParsingSettings,
    PipelineSettings,
    RuntimeSettings,
    TranscriptionSettings,
    ValidationSettings,
)
from receipt_intelligence.extraction.transcription import (
    build_canonical_transcription_service,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("var/next_transcription"))
    parser.add_argument("--run-id", default="single-transcription")
    parser.add_argument("--ollama-url", default="http://localhost:11434")
    parser.add_argument("--qwen-model", default="qwen3.5:4b")
    parser.add_argument("--paddle-device", default="cpu")
    parser.add_argument("--paddle-lang", default="en")
    parser.add_argument("--max-crops", type=int, default=4)
    parser.add_argument("--parallelism", type=int, default=1)
    parser.add_argument("--retries", type=int, default=1)
    parser.add_argument("--overwrite", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    settings = PipelineSettings(
        source_image_path=args.image,
        legacy_ocr_json_path=None,
        detection=DetectionSettings(language=args.paddle_lang, device=args.paddle_device),
        transcription=TranscriptionSettings(
            ollama_url=args.ollama_url,
            model=args.qwen_model,
            parallelism=args.parallelism,
            retries=args.retries,
        ),
        parsing=ParsingSettings(ollama_url=args.ollama_url, model="unused-in-phase-2"),
        validation=ValidationSettings(),
        correction=CorrectionSettings(enabled=False),
        categorization=CategorizationSettings(enabled=False),
        runtime=RuntimeSettings(result_dir=args.output_dir, run_id=args.run_id),
        crop_planning=CropPlanningSettings(max_crops=args.max_crops),
    )
    service = build_canonical_transcription_service(settings, overwrite=args.overwrite)
    result = service.transcribe(
        TranscriptionRequest(source_image_path=args.image, run_id=args.run_id)
    )
    print(result.canonical_text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
