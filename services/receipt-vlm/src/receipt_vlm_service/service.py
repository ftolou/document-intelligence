"""Image preparation and PaddleOCR-VL CLI orchestration."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

try:
    from PIL import Image
except Exception:  # pragma: no cover
    Image = None  # type: ignore[assignment]

from receipt_vlm_service.cli import run_paddle_cli
from receipt_vlm_service.json_values import jsonable, save_json
from receipt_vlm_service.settings import VlmSettings


def prepare_image(
    image_path: Path,
    result_dir: Path,
    max_side: int,
) -> tuple[Path, dict[str, Any]]:
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
        metadata.update({"resized": False, "error": f"{type(exc).__name__}: {exc}"})
        return image_path, metadata


class VlmService:
    def __init__(self, settings: VlmSettings) -> None:
        self.settings = settings

    def execute(self, *, image_path: Path, run_id: str) -> dict[str, Any]:
        started = time.perf_counter()
        result_dir = self.settings.results_dir / run_id
        result_dir.mkdir(parents=True, exist_ok=True)
        output_path = result_dir / f"{run_id}_vlm_service_raw.json"

        if self.settings.backend not in {
            "paddleocr_vl",
            "paddleocr-vl",
            "paddleocrvl",
            "local",
        }:
            result: dict[str, Any] = {
                "status": "error",
                "backend": self.settings.backend,
                "error": f"Unsupported VLM service backend: {self.settings.backend}",
            }
        elif self.settings.runner not in {"cli", "doc_parser", "paddleocr_cli"}:
            result = {
                "status": "error",
                "backend": "paddleocr_vl",
                "error": (
                    "The standalone VLM image supports only the PaddleOCR doc_parser CLI; "
                    f"configured runner: {self.settings.runner}"
                ),
            }
        else:
            prepared_path, preparation = prepare_image(
                image_path,
                result_dir,
                self.settings.max_side_limit,
            )
            result = run_paddle_cli(
                prepared_path,
                result_dir,
                self.settings.timeout_seconds,
                device=self.settings.device,
                engine=self.settings.engine,
            )
            result.update(
                {
                    "image_path": str(image_path),
                    "prepared_image_path": str(prepared_path),
                    "image_prepare": preparation,
                }
            )

        result.update(
            {
                "service_version": self.settings.app_version,
                "runner": self.settings.runner,
                "engine": self.settings.engine,
                "device": self.settings.device,
                "run_id": run_id,
            }
        )
        result.setdefault("duration_seconds", round(time.perf_counter() - started, 2))
        normalized = jsonable(result)
        assert isinstance(normalized, dict)
        save_json(output_path, normalized)
        return normalized


__all__ = ["VlmService", "prepare_image"]
