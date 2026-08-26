#!/usr/bin/env python3
"""Canonical typed receipt extraction entry point."""

from __future__ import annotations

import time
from typing import Any

from receipt_intelligence.extraction.composition import build_extraction_dependencies
from receipt_intelligence.extraction.config import ExtractionRequest
from receipt_intelligence.extraction.context import ExtractionContext
from receipt_intelligence.extraction.dependencies import ExtractionDependencies
from receipt_intelligence.extraction.factory import build_extraction_workflow
from receipt_intelligence.extraction.source_image import validate_source_image
from receipt_intelligence.observability.timing import utc_now_iso


def run_receipt_extraction(
    request: ExtractionRequest,
    *,
    dependencies: ExtractionDependencies | None = None,
) -> dict[str, Any]:
    """Run one receipt image through the selected extraction backend."""

    validate_source_image(
        request.source_image_path,
        max_width=request.source_image_max_width,
        max_height=request.source_image_max_height,
        max_pixels=request.source_image_max_pixels,
    )

    if request.extraction_backend == "openai_one_shot":
        from receipt_intelligence.adapters.llm import OpenAIMultimodalGateway
        from receipt_intelligence.extraction.openai_observability import (
            build_observed_openai_client,
            publish_openai_extraction_metrics,
        )
        from receipt_intelligence.extraction.openai_one_shot import (
            run_openai_one_shot_extraction,
        )

        started_at = utc_now_iso()
        started = time.perf_counter()
        try:
            client = build_observed_openai_client(request)
            result = run_openai_one_shot_extraction(
                request,
                gateway=OpenAIMultimodalGateway(
                    client=client,
                    reasoning_effort=request.openai_reasoning_effort,
                    image_detail=request.openai_image_detail,
                ),
            )
        except Exception as exc:
            publish_openai_extraction_metrics(
                request,
                status="failed",
                started_at=started_at,
                duration_ms=(time.perf_counter() - started) * 1000.0,
                stages=(),
                error=f"{type(exc).__name__}: {exc}",
            )
            raise

        observability = (
            result.get("observability") if isinstance(result.get("observability"), dict) else {}
        )
        stage_trace = observability.get("stage_trace")
        stages = (
            tuple(value for value in stage_trace if isinstance(value, dict))
            if isinstance(stage_trace, list)
            else ()
        )
        metrics_path = publish_openai_extraction_metrics(
            request,
            status="completed",
            started_at=started_at,
            duration_ms=(time.perf_counter() - started) * 1000.0,
            stages=stages,
        )
        paths = dict(result.get("paths") or {})
        paths["extraction_metrics"] = metrics_path
        result["paths"] = paths
        result["observability"] = {
            **observability,
            "stage_trace": [dict(value) for value in stages],
            "metrics_path": metrics_path,
        }
        return result

    context = ExtractionContext(
        config=request,
        dependencies=dependencies or build_extraction_dependencies(request),
    )
    completed = build_extraction_workflow().run(context)
    return _application_result(completed)


def _application_result(context: ExtractionContext) -> dict[str, Any]:
    finalized = context.require_finalized().result
    if finalized is None:
        raise RuntimeError("Extraction workflow finished without a finalization result.")
    result = finalized.as_application_result()
    result["paths"] = dict(context.paths)
    result["logs"] = list(context.logs)
    result["observability"] = {
        "stage_trace": [dict(entry) for entry in context.stage_trace],
        "metrics_path": context.paths.get("extraction_metrics"),
    }
    return result


__all__ = ["run_receipt_extraction"]
