"""Use case for analyzing one image with a configured visual-model engine."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

try:
    from PIL import Image
except Exception:  # pragma: no cover - resizing is optional
    Image = None  # type: ignore[assignment]

from receipt_intelligence.application.ports.vlm import VlmEngine, VlmRequest
from receipt_intelligence.runtime.json_values import jsonable, save_json


def prepare_image_for_vlm(
    image_path: Path,
    result_dir: Path,
    max_side: int,
) -> tuple[Path, dict[str, Any]]:
    """Create a bounded copy for VLM processing while preserving the source image."""
    metadata: dict[str, Any] = {
        "original_path": str(image_path),
        "max_side_limit": max_side,
    }
    if max_side <= 0:
        metadata.update({"resized": False, "reason": "disabled"})
        return image_path, metadata
    if Image is None:
        metadata.update({"resized": False, "reason": "pillow_unavailable"})
        return image_path, metadata

    try:
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            width, height = image.size
            metadata.update({"original_width": width, "original_height": height})
            side = max(width, height)
            if side <= max_side:
                metadata.update(
                    {
                        "resized": False,
                        "prepared_width": width,
                        "prepared_height": height,
                    }
                )
                return image_path, metadata

            scale = max_side / float(side)
            prepared_width = max(1, int(round(width * scale)))
            prepared_height = max(1, int(round(height * scale)))
            prepared_path = result_dir / f"{image_path.stem}_vlm_resized_{max_side}.jpg"
            image = image.resize((prepared_width, prepared_height))
            image.save(prepared_path, quality=92, optimize=True)
            metadata.update(
                {
                    "resized": True,
                    "prepared_path": str(prepared_path),
                    "prepared_width": prepared_width,
                    "prepared_height": prepared_height,
                    "scale": scale,
                }
            )
            return prepared_path, metadata
    except Exception as exc:
        metadata.update(
            {
                "resized": False,
                "error": f"{type(exc).__name__}: {exc}",
            }
        )
        return image_path, metadata


class VlmAnalysisService:
    """Coordinate image preparation, adapter invocation, and result persistence."""

    def __init__(
        self,
        *,
        engine: VlmEngine,
        results_dir: Path,
        service_version: str,
        timeout_seconds: float,
        max_side_limit: int,
        runner_name: str,
        engine_name: str,
        device_name: str,
    ) -> None:
        self.engine = engine
        self.results_dir = Path(results_dir)
        self.service_version = service_version
        self.timeout_seconds = timeout_seconds
        self.max_side_limit = max_side_limit
        self.runner_name = runner_name
        self.engine_name = engine_name
        self.device_name = device_name

    def execute(self, *, image_path: Path, run_id: str) -> dict[str, Any]:
        started = time.perf_counter()
        result_dir = self.results_dir / run_id
        result_dir.mkdir(parents=True, exist_ok=True)
        output_path = result_dir / f"{run_id}_vlm_service_raw.json"

        prepared_path, preparation = prepare_image_for_vlm(
            image_path,
            result_dir,
            self.max_side_limit,
        )
        result = dict(
            self.engine.analyze(
                VlmRequest(
                    image_path=prepared_path,
                    result_dir=result_dir,
                    run_id=run_id,
                    enabled=True,
                    timeout_seconds=self.timeout_seconds,
                )
            )
        )
        result.update(
            {
                "service_version": self.service_version,
                "image_path": str(image_path),
                "prepared_image_path": str(prepared_path),
                "image_prepare": preparation,
                "runner": self.runner_name,
                "engine": self.engine_name,
                "device": self.device_name,
                "run_id": run_id,
            }
        )
        result.setdefault(
            "duration_seconds",
            round(time.perf_counter() - started, 2),
        )
        normalized = jsonable(result)
        assert isinstance(normalized, dict)
        save_json(output_path, normalized)
        return normalized


__all__ = ["VlmAnalysisService", "prepare_image_for_vlm"]
