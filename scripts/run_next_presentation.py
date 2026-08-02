#!/usr/bin/env python3
"""Run Phase 6 categorization/finalization on a corrected receipt artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from receipt_intelligence.adapters.llm import OllamaGateway
from receipt_intelligence.app_version import get_app_version
from receipt_intelligence.extraction.contracts.presentation import (
    CategorizationRequest,
    FinalizationRequest,
)
from receipt_intelligence.extraction.contracts.validation import ValidationReport
from receipt_intelligence.extraction.presentation import (
    CompatibilityFilesystemArtifactStore,
    CompatibilityFinalizationService,
    ExistingReceiptCategorizationService,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Categorize and finalize an already corrected next-pipeline receipt."
    )
    parser.add_argument("receipt_json", type=Path)
    parser.add_argument("validation_json", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--ollama-url", default="http://localhost:11434")
    parser.add_argument("--model", default="gemma4:latest")
    parser.add_argument("--num-ctx", type=int, default=8192)
    parser.add_argument("--num-predict", type=int, default=4096)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--keep-alive", default="")
    parser.add_argument("--disable-categorization", action="store_true")
    args = parser.parse_args()

    receipt = json.loads(args.receipt_json.read_text(encoding="utf-8-sig"))
    validation_raw = json.loads(args.validation_json.read_text(encoding="utf-8-sig"))
    validation = ValidationReport.from_legacy(validation_raw)
    categorizer = ExistingReceiptCategorizationService(
        llm_gateway=OllamaGateway(args.ollama_url),
        ollama_url=args.ollama_url,
        model=args.model,
        num_ctx=args.num_ctx,
        num_predict=args.num_predict,
        keep_alive=args.keep_alive or None,
        timeout_seconds=args.timeout,
    )
    categorization = categorizer.categorize(
        CategorizationRequest(
            run_id=args.run_id,
            receipt=receipt,
            enabled=not args.disable_categorization,
        )
    )
    finalizer = CompatibilityFinalizationService(
        artifact_store=CompatibilityFilesystemArtifactStore(args.out_dir),
        app_version=get_app_version(),
        overwrite=True,
    )
    result = finalizer.finalize(
        FinalizationRequest(
            run_id=args.run_id,
            receipt=receipt,
            validation=validation,
            categorization=categorization,
            upstream_metadata={"mode": "standalone_phase6_presentation"},
        )
    )
    print(
        json.dumps(
            {
                "status": result.validation.status,
                "categorization_status": result.categorization.status.value,
                "paths": {key: str(value) for key, value in result.paths.items()},
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
