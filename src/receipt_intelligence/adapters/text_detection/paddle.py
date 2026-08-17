"""PaddleOCR implementation of provider-neutral text detection."""

from __future__ import annotations

import inspect
import json
import time
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps

from receipt_intelligence.application.ports.text_detection import (
    DetectedTextRegion,
    TextDetectionEngine,
    TextDetectionRequest,
    TextDetectionResult,
)


class PaddleTextDetectionEngine(TextDetectionEngine):
    """Use Paddle only for geometry; recognized text is deliberately ignored."""

    def __init__(self, *, backend: str = "auto", model_name: str | None = None) -> None:
        normalized = str(backend or "").strip().lower()
        if normalized not in {"auto", "text_detection", "paddleocr"}:
            raise ValueError(f"Unsupported Paddle detection backend: {backend!r}")
        self.backend = normalized
        self.model_name = str(model_name).strip() if model_name else None
        self._cache: dict[tuple[str, str | None, str, str], tuple[str, Any]] = {}

    def detect(self, request: TextDetectionRequest) -> TextDetectionResult:
        with Image.open(request.image_path) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
        backend, engine = self._initialize(request)
        started = time.perf_counter()
        raw_result = _predict(backend, engine, request.image_path, image)
        duration_ms = (time.perf_counter() - started) * 1000.0
        polygons, scores = _extract_polygons_and_scores(raw_result)
        if not polygons:
            raise RuntimeError("PaddleOCR returned no parseable detection polygons.")
        regions = tuple(
            DetectedTextRegion(
                region_id=f"D{index:04d}",
                polygon=polygon,
                score=score,
            )
            for index, (polygon, score) in enumerate(zip(polygons, scores, strict=True), start=1)
        )
        return TextDetectionResult(
            regions=regions,
            image_width=image.width,
            image_height=image.height,
            duration_ms=duration_ms,
            diagnostics={
                "backend": backend,
                "model_name": self.model_name,
                "device": request.device,
                "language": request.language,
                "raw_polygon_count": len(polygons),
            },
        )

    def _initialize(self, request: TextDetectionRequest) -> tuple[str, Any]:
        cache_key = (self.backend, self.model_name, request.device, request.language)
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached
        errors: list[str] = []

        if self.backend in {"auto", "text_detection"}:
            try:
                from paddleocr import TextDetection

                engine = TextDetection(
                    **_filtered_kwargs(
                        TextDetection,
                        {"model_name": self.model_name, "device": request.device},
                    )
                )
                result = ("text_detection", engine)
                self._cache[cache_key] = result
                return result
            except Exception as exc:
                errors.append(f"TextDetection initialization failed: {type(exc).__name__}: {exc}")
                if self.backend == "text_detection":
                    raise RuntimeError("; ".join(errors)) from exc

        if self.backend in {"auto", "paddleocr"}:
            try:
                from paddleocr import PaddleOCR

                variants = [
                    {
                        "lang": request.language,
                        "device": request.device,
                        "use_doc_orientation_classify": False,
                        "use_doc_unwarping": False,
                        "use_textline_orientation": False,
                        "show_log": False,
                    },
                    {
                        "lang": request.language,
                        "use_angle_cls": False,
                        "use_gpu": request.device.casefold().startswith("gpu"),
                        "show_log": False,
                    },
                    {"lang": request.language},
                    {},
                ]
                last_error: Exception | None = None
                for candidate in variants:
                    try:
                        engine = PaddleOCR(**_filtered_kwargs(PaddleOCR, candidate))
                        result = ("paddleocr", engine)
                        self._cache[cache_key] = result
                        return result
                    except Exception as exc:
                        last_error = exc
                if last_error is not None:
                    raise last_error
            except Exception as exc:
                errors.append(f"PaddleOCR initialization failed: {type(exc).__name__}: {exc}")
        raise RuntimeError("Could not initialize a PaddleOCR detector. " + "; ".join(errors))


def _predict(backend: str, engine: Any, image_path: Path, image: Image.Image) -> Any:
    errors: list[str] = []
    if backend == "text_detection":
        predict = getattr(engine, "predict", None)
        if not callable(predict):
            raise RuntimeError("TextDetection engine has no predict() method.")
        for kwargs in (
            {"input": str(image_path), "batch_size": 1},
            {"input": str(image_path)},
            {"input": image},
        ):
            try:
                result = predict(**_filtered_kwargs(predict, kwargs))
                return list(result) if not isinstance(result, list) else result
            except Exception as exc:
                errors.append(f"TextDetection.predict failed: {type(exc).__name__}: {exc}")
        raise RuntimeError("; ".join(errors))

    legacy_ocr = getattr(engine, "ocr", None)
    if callable(legacy_ocr):
        try:
            import numpy as np

            return legacy_ocr(np.asarray(image), det=True, rec=False, cls=False)
        except Exception as exc:
            errors.append(f"PaddleOCR.ocr detection failed: {type(exc).__name__}: {exc}")
    predict = getattr(engine, "predict", None)
    if callable(predict):
        for kwargs in ({"input": str(image_path)}, {"input": image}):
            try:
                result = predict(**_filtered_kwargs(predict, kwargs))
                return list(result) if not isinstance(result, list) else result
            except Exception as exc:
                errors.append(f"PaddleOCR.predict failed: {type(exc).__name__}: {exc}")
    raise RuntimeError("PaddleOCR detector call failed. " + "; ".join(errors))


def _filtered_kwargs(callable_object: Any, values: dict[str, Any]) -> dict[str, Any]:
    try:
        signature = inspect.signature(callable_object)
    except (TypeError, ValueError):
        return {key: value for key, value in values.items() if value is not None}
    if any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    ):
        return {key: value for key, value in values.items() if value is not None}
    return {
        key: value
        for key, value in values.items()
        if key in signature.parameters and value is not None
    }


def _to_plain(value: Any) -> Any:
    if hasattr(value, "tolist"):
        try:
            return value.tolist()
        except Exception:
            pass
    return value


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _is_point(value: Any) -> bool:
    value = _to_plain(value)
    return (
        isinstance(value, (list, tuple))
        and len(value) >= 2
        and _is_number(value[0])
        and _is_number(value[1])
    )


def _coerce_polygon(value: Any) -> tuple[tuple[float, float], ...] | None:
    value = _to_plain(value)
    if (
        isinstance(value, (list, tuple))
        and len(value) == 4
        and all(_is_number(item) for item in value)
    ):
        x_min, y_min, x_max, y_max = (float(item) for item in value)
        return ((x_min, y_min), (x_max, y_min), (x_max, y_max), (x_min, y_max))
    if not isinstance(value, (list, tuple)) or len(value) < 4:
        return None
    if not all(_is_point(point) for point in value):
        return None
    return tuple((float(_to_plain(point)[0]), float(_to_plain(point)[1])) for point in value)


def _mapping(value: Any) -> dict[str, Any] | None:
    if isinstance(value, dict):
        return value
    for attribute_name in ("json", "to_dict", "dict"):
        attribute = getattr(value, attribute_name, None)
        try:
            candidate = attribute() if callable(attribute) else attribute
        except Exception:
            continue
        if isinstance(candidate, str):
            try:
                candidate = json.loads(candidate)
            except Exception:
                continue
        if isinstance(candidate, dict):
            return candidate
    raw = getattr(value, "__dict__", None)
    return raw if isinstance(raw, dict) else None


def _find_named(value: Any, keys: set[str]) -> Any:
    value = _to_plain(value)
    mapping = _mapping(value)
    if mapping is not None:
        for key, candidate in mapping.items():
            if str(key).casefold() in keys:
                return candidate
        for candidate in mapping.values():
            found = _find_named(candidate, keys)
            if found is not None:
                return found
        return None
    if isinstance(value, (list, tuple)):
        for candidate in value:
            found = _find_named(candidate, keys)
            if found is not None:
                return found
    return None


def _collect_polygons(value: Any) -> list[tuple[tuple[float, float], ...]]:
    value = _to_plain(value)
    polygon = _coerce_polygon(value)
    if polygon is not None:
        return [polygon]
    mapping = _mapping(value)
    if mapping is not None:
        collected: list[tuple[tuple[float, float], ...]] = []
        for candidate in mapping.values():
            collected.extend(_collect_polygons(candidate))
        return collected
    if isinstance(value, (list, tuple)):
        collected = []
        for candidate in value:
            collected.extend(_collect_polygons(candidate))
        return collected
    return []


def _extract_polygons_and_scores(
    raw_result: Any,
) -> tuple[list[tuple[tuple[float, float], ...]], list[float | None]]:
    polygon_value = _find_named(
        raw_result,
        {"dt_polys", "det_polys", "rec_polys", "polys", "boxes", "text_boxes"},
    )
    polygons = _collect_polygons(polygon_value if polygon_value is not None else raw_result)
    unique: list[tuple[tuple[float, float], ...]] = []
    seen: set[tuple[tuple[int, int], ...]] = set()
    for polygon in polygons:
        key = tuple((round(x), round(y)) for x, y in polygon)
        if key not in seen:
            seen.add(key)
            unique.append(polygon)
    score_value = _find_named(raw_result, {"dt_scores", "det_scores", "scores", "text_scores"})
    raw_scores = _to_plain(score_value)
    scores: list[float | None] = []
    if isinstance(raw_scores, (list, tuple)):
        scores.extend(
            float(score) if _is_number(_to_plain(score)) else None for score in raw_scores
        )
    scores.extend([None] * max(0, len(unique) - len(scores)))
    return unique, scores[: len(unique)]


__all__ = ["PaddleTextDetectionEngine"]
