"""Canonical Paddle-geometry/Qwen transcription service."""

from __future__ import annotations

import concurrent.futures
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFilter, ImageOps

from receipt_intelligence.application.ports.multimodal import (
    MultimodalGateway,
    MultimodalGenerationRequest,
)
from receipt_intelligence.application.ports.text_detection import (
    TextDetectionEngine,
    TextDetectionRequest,
)
from receipt_intelligence.extraction.contracts.common import StageArtifact
from receipt_intelligence.extraction.contracts.transcription import (
    ReceiptCrop,
    TranscriptionFragment,
    TranscriptionRequest,
    TranscriptionResult,
)
from receipt_intelligence.extraction.services.transcription import TranscriptionService
from receipt_intelligence.extraction.settings import (
    CropPlanningSettings,
    DetectionSettings,
    TranscriptionSettings,
)
from receipt_intelligence.extraction.transcription.canonical import (
    build_canonical_rows,
    clean_plain_lines,
    serialize_canonical_rows,
)
from receipt_intelligence.extraction.transcription.crop_planner import (
    full_image_crop,
    plan_safe_crops,
)
from receipt_intelligence.extraction.transcription.line_clustering import cluster_text_regions
from receipt_intelligence.extraction.transcription.models import CropPlan, CropSpec, DetectedLine
from receipt_intelligence.prompts.registry import PromptReference, PromptRegistry

_QWEN_PROMPT = PromptReference("qwen.transcription", "1.0.0")


class CanonicalReceiptTranscriptionService(TranscriptionService):
    """Create one ordered, canonical transcription from Paddle geometry and Qwen text.

    Paddle text is never used. Qwen output receives only transport-level cleanup: blank lines,
    code fences, bullets, and accidental row-id wrappers are removed. There is intentionally no
    duplication, line-count, semantic, or arithmetic validation after transcription.
    """

    def __init__(
        self,
        *,
        detector: TextDetectionEngine,
        multimodal_gateway: MultimodalGateway,
        prompt_registry: PromptRegistry,
        result_dir: Path,
        detection_settings: DetectionSettings,
        crop_settings: CropPlanningSettings,
        transcription_settings: TranscriptionSettings,
        overwrite: bool = True,
    ) -> None:
        self._detector = detector
        self._gateway = multimodal_gateway
        self._prompts = prompt_registry
        self._result_dir = Path(result_dir)
        self._detection = detection_settings
        self._crops = crop_settings
        self._transcription = transcription_settings
        self._overwrite = overwrite

    def transcribe(self, request: TranscriptionRequest) -> TranscriptionResult:
        started = time.perf_counter()
        work_dir = self._result_dir / f"{request.run_id}_transcription"
        work_dir.mkdir(parents=True, exist_ok=True)
        with Image.open(request.source_image_path) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")

        detected_lines: tuple[DetectedLine, ...] = ()
        detector_diagnostics: dict[str, Any] = {}
        detector_error: dict[str, Any] | None = None
        try:
            detected = self._detector.detect(
                TextDetectionRequest(
                    image_path=request.source_image_path,
                    language=self._detection.language,
                    device=self._detection.device,
                    max_side_length=self._detection.max_side_length,
                )
            )
            detected_lines = cluster_text_regions(detected.regions, self._detection)
            detector_diagnostics = {
                **detected.diagnostics,
                "duration_ms": detected.duration_ms,
                "raw_region_count": len(detected.regions),
                "detected_line_count": len(detected_lines),
            }
        except Exception as exc:
            detector_error = {
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
            detector_diagnostics = {"status": "error", **detector_error}

        self._write_json(
            work_dir / "10_paddle_line_boxes.json",
            {
                "detector": detector_diagnostics,
                "error": detector_error,
                "lines": [_line_payload(line) for line in detected_lines],
            },
        )
        _save_overlay(work_dir / "10_paddle_line_overlay.png", image, detected_lines)

        plan = self._build_plan(image, detected_lines, detector_error)
        self._write_json(work_dir / "10_qwen_safe_cut_groups.json", _plan_payload(plan))

        accepted, attempts, diagnostics = self._run_plan(plan, work_dir)
        runtime_fallback = False
        if len(accepted) != len(plan.crops) and not plan.crops[0].crop_id.startswith("GFULL"):
            runtime_fallback = True
            diagnostics.append(
                {
                    "status": "discarded_partial_crop_transcription",
                    "reason": "one_or_more_crop_calls_failed",
                }
            )
            fallback = full_image_crop(
                image,
                detected_lines,
                crop_id="GFULL_RUNTIME",
                fallback=True,
            )
            accepted, fallback_attempts, fallback_diagnostics = self._run_plan(
                CropPlan(
                    crops=(fallback,),
                    boundaries=(),
                    boundary_decisions=(),
                    metadata={
                        "status": "runtime_full_image_fallback",
                        "effective_crops": 1,
                    },
                ),
                work_dir,
            )
            attempts.extend(fallback_attempts)
            diagnostics.extend(fallback_diagnostics)

        if not accepted:
            raise RuntimeError(
                "Qwen produced no nonempty transcription, including whole-image fallback."
            )

        accepted.sort(key=lambda item: item[0].order)
        fragments = tuple(item[0] for item in accepted)
        crops = tuple(item[1] for item in accepted)
        rows = build_canonical_rows(fragments)
        canonical_text = serialize_canonical_rows(rows)

        transcription_path = work_dir / "11_transcription.txt"
        transcription_path.write_text(canonical_text + "\n", encoding="utf-8")
        report_path = work_dir / "10_qwen_transcription.json"
        report = {
            "status": "completed",
            "model": self._transcription.model,
            "image": str(request.source_image_path),
            "row_count": len(rows),
            "method": (
                "single_whole_image_qwen_transcription"
                if any(crop.crop_id.startswith("GFULL") for crop in crops)
                else "ordered_non_overlapping_crop_concatenation"
            ),
            "post_transcription_validation_used": False,
            "paddle_text_used": False,
            "matching_used": False,
            "runtime_full_image_fallback": runtime_fallback,
            "crop_plan": plan.metadata,
            "attempts": attempts,
            "group_diagnostics": diagnostics,
            "duration_seconds": round(time.perf_counter() - started, 3),
            "metrics": _aggregate_metrics(
                fragments,
                detector_diagnostics,
                wall_duration_seconds=time.perf_counter() - started,
            ),
        }
        self._write_json(report_path, report)

        return TranscriptionResult(
            canonical_text=canonical_text,
            rows=rows,
            crops=crops,
            fragments=fragments,
            diagnostics=report,
            artifacts=(
                StageArtifact(
                    name="transcription",
                    path=transcription_path,
                    media_type="text/plain",
                ),
                StageArtifact(name="transcription_report", path=report_path),
            ),
        )

    def _build_plan(
        self,
        image: Image.Image,
        lines: tuple[DetectedLine, ...],
        detector_error: dict[str, Any] | None,
    ) -> CropPlan:
        fallback_reason: str | None = None
        if detector_error is not None:
            fallback_reason = "paddle_detection_failed"
        elif len(lines) < self._detection.minimum_lines:
            fallback_reason = "too_few_detected_lines_for_safe_cropping"
        elif len(lines) > self._detection.maximum_lines:
            fallback_reason = "too_many_detected_lines_for_safe_cropping"

        if fallback_reason is None:
            try:
                return plan_safe_crops(image, lines, self._crops)
            except Exception as exc:
                fallback_reason = f"safe_crop_planning_failed:{type(exc).__name__}:{exc}"

        crop = full_image_crop(image, lines, fallback=True)
        return CropPlan(
            crops=(crop,),
            boundaries=(),
            boundary_decisions=(),
            metadata={
                "requested_crops": self._crops.max_crops,
                "initial_effective_crops": 1,
                "effective_crops": 1,
                "status": "fallback_full_image",
                "fallback_reason": fallback_reason,
                "image_width": image.width,
                "image_height": image.height,
                "image_aspect_ratio_h_over_w": round(
                    float(image.height) / float(max(1, image.width)),
                    6,
                ),
            },
        )

    def _run_plan(
        self,
        plan: CropPlan,
        work_dir: Path,
    ) -> tuple[
        list[tuple[TranscriptionFragment, ReceiptCrop]],
        list[dict[str, Any]],
        list[dict[str, Any]],
    ]:
        max_workers = min(len(plan.crops), max(1, self._transcription.parallelism))
        completed: list[tuple[TranscriptionFragment | None, ReceiptCrop, list[dict[str, Any]]]] = []
        if max_workers == 1:
            completed = [
                self._transcribe_crop(crop, index, work_dir)
                for index, crop in enumerate(plan.crops)
            ]
        else:
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(self._transcribe_crop, crop, index, work_dir): crop
                    for index, crop in enumerate(plan.crops)
                }
                for future in concurrent.futures.as_completed(futures):
                    completed.append(future.result())

        accepted: list[tuple[TranscriptionFragment, ReceiptCrop]] = []
        attempts: list[dict[str, Any]] = []
        diagnostics: list[dict[str, Any]] = []
        for fragment, crop, crop_attempts in completed:
            attempts.extend(crop_attempts)
            if fragment is None:
                diagnostics.append(
                    {
                        "crop_id": crop.crop_id,
                        "status": "no_nonempty_qwen_response",
                        "bbox": list(crop.source_box),
                    }
                )
                continue
            accepted.append((fragment, crop))
            diagnostics.append(
                {
                    "crop_id": crop.crop_id,
                    "status": "accepted_without_post_transcription_validation",
                    "paddle_detected_line_estimate": len(crop.detected_line_indices),
                    "qwen_returned_line_count": len(clean_plain_lines(fragment.text)),
                    "transcription_text_source": fragment.text_source,
                    "bbox": list(crop.source_box),
                    "attempt": fragment.attempt,
                }
            )
        return accepted, attempts, diagnostics

    def _transcribe_crop(
        self,
        spec: CropSpec,
        order: int,
        work_dir: Path,
    ) -> tuple[TranscriptionFragment | None, ReceiptCrop, list[dict[str, Any]]]:
        crop_path = work_dir / f"group_{spec.crop_id}.png"
        processed = _preprocess_crop(
            spec.image,
            scale=self._transcription.crop_scale,
            sharpen=self._transcription.crop_sharpen,
        )
        if self._overwrite or not crop_path.exists():
            processed.save(crop_path, format="PNG")
        crop = ReceiptCrop(
            crop_id=spec.crop_id,
            image_path=crop_path,
            source_box=(float(spec.left), float(spec.top), float(spec.right), float(spec.bottom)),
            order=order,
            is_full_image_fallback=spec.is_full_image_fallback,
            detected_line_indices=spec.line_indices,
        )
        attempts: list[dict[str, Any]] = []
        for attempt in range(1, self._transcription.retries + 2):
            try:
                result = self._gateway.generate(
                    MultimodalGenerationRequest(
                        model=self._transcription.model,
                        prompt=self._prompts.read_template(_QWEN_PROMPT),
                        image_paths=(crop_path,),
                        operation="receipt_transcription",
                        attempt=attempt,
                        think=self._transcription.think,
                        num_ctx=self._transcription.num_ctx,
                        num_predict=(
                            self._transcription.num_predict
                            if spec.crop_id.startswith("GFULL")
                            else min(
                                self._transcription.num_predict,
                                max(256, len(spec.line_indices) * 160),
                            )
                        ),
                        temperature=self._transcription.temperature,
                        keep_alive=self._transcription.keep_alive,
                        timeout_seconds=self._transcription.timeout_seconds,
                    )
                )
                clean_plain_lines(result.text)
                raw_path = work_dir / f"group_{spec.crop_id}_attempt_{attempt:02d}_response.json"
                self._write_json(raw_path, result.raw_response or {"text": result.text})
                attempts.append(
                    {
                        "crop_id": spec.crop_id,
                        "attempt": attempt,
                        "status": "accepted",
                        "text_source": result.text_source,
                        "metrics": (
                            result.metrics.to_diagnostics() if result.metrics is not None else None
                        ),
                    }
                )
                return (
                    TranscriptionFragment(
                        crop_id=spec.crop_id,
                        text=result.text,
                        order=order,
                        metrics=result.metrics,
                        attempt=attempt,
                        text_source=result.text_source,
                    ),
                    crop,
                    attempts,
                )
            except Exception as exc:
                attempts.append(
                    {
                        "crop_id": spec.crop_id,
                        "attempt": attempt,
                        "status": "qwen_call_error",
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                )
        return None, crop, attempts

    @staticmethod
    def _write_json(path: Path, payload: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default),
            encoding="utf-8",
        )


def _preprocess_crop(crop: Image.Image, *, scale: float, sharpen: bool) -> Image.Image:
    processed = ImageOps.exif_transpose(crop).convert("L")
    if sharpen:
        processed = processed.filter(ImageFilter.SHARPEN)
    if scale != 1.0:
        processed = processed.resize(
            (
                max(1, round(processed.width * scale)),
                max(1, round(processed.height * scale)),
            ),
            Image.Resampling.LANCZOS,
        )
    return processed.convert("RGB")


def _save_overlay(path: Path, image: Image.Image, lines: tuple[DetectedLine, ...]) -> None:
    overlay = image.copy()
    draw = ImageDraw.Draw(overlay)
    for line in lines:
        draw.rectangle((line.x_min, line.y_min, line.x_max, line.y_max), width=2)
        draw.text((line.x_min, max(0, line.y_min - 12)), f"L{line.index + 1:04d}")
    overlay.save(path, format="PNG")


def _line_payload(line: DetectedLine) -> dict[str, Any]:
    return {
        "line_id": f"L{line.index + 1:04d}",
        "line_index": line.index,
        "region_ids": list(line.region_ids),
        "bbox": [line.x_min, line.y_min, line.x_max, line.y_max],
        "center_y": line.center_y,
    }


def _plan_payload(plan: CropPlan) -> dict[str, Any]:
    return {
        "method": "aspect_ratio_adaptive_paddle_snapped_or_full_image_fallback",
        "paddle_text_used": False,
        "matching_used": False,
        "post_transcription_validation_used": False,
        "non_overlapping": True,
        "crop_plan": plan.metadata,
        "boundary_decisions": list(plan.boundary_decisions),
        "groups": [
            {
                "crop_id": crop.crop_id,
                "line_indices": list(crop.line_indices),
                "bbox": [crop.left, crop.top, crop.right, crop.bottom],
                "full_image_fallback": crop.is_full_image_fallback,
            }
            for crop in plan.crops
        ],
    }


def _aggregate_metrics(
    fragments: tuple[TranscriptionFragment, ...],
    detector_diagnostics: dict[str, Any],
    *,
    wall_duration_seconds: float,
) -> dict[str, Any]:
    metrics = [fragment.metrics for fragment in fragments if fragment.metrics is not None]

    def total(field: str) -> int | float | None:
        values = [
            getattr(metric, field) for metric in metrics if getattr(metric, field) is not None
        ]
        return sum(values) if values else None

    return {
        "detector_duration_ms": detector_diagnostics.get("duration_ms"),
        "qwen_call_count": len(fragments),
        "prompt_eval_count": total("prompt_eval_count"),
        "eval_count": total("eval_count"),
        "total_duration_ns": total("total_duration_ns"),
        "load_duration_ns": total("load_duration_ns"),
        "prompt_eval_duration_ns": total("prompt_eval_duration_ns"),
        "eval_duration_ns": total("eval_duration_ns"),
        "done_reasons": [metric.done_reason for metric in metrics],
        "wall_duration_seconds": round(wall_duration_seconds, 3),
    }


def _json_default(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if hasattr(value, "__dataclass_fields__"):
        return asdict(value)
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


__all__ = ["CanonicalReceiptTranscriptionService"]
