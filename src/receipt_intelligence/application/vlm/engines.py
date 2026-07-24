"""Application policies that compose visual-model adapters."""

from __future__ import annotations

from typing import Any

from receipt_intelligence.application.ports.vlm import VlmEngine, VlmRequest
from receipt_intelligence.runtime.json_values import save_json


def _emit(
    request: VlmRequest,
    status: str,
    message: str,
    **details: Any,
) -> None:
    callback = request.progress_callback
    if callback is None:
        return
    try:
        callback(
            {
                "stage": "visual_evidence",
                "status": status,
                "message": message,
                "details": details,
                "source": "vlm_orchestrator",
            }
        )
    except Exception:
        pass


class DisabledVlmEngine(VlmEngine):
    """Explicit no-op engine used when no VLM backend is configured."""

    def __init__(self, message: str = "VLM evidence is disabled by configuration.") -> None:
        self.message = message

    def analyze(self, request: VlmRequest) -> dict[str, Any]:
        return {"status": "disabled", "backend": "disabled", "message": self.message}


class UnsupportedVlmEngine(VlmEngine):
    """Return a structured result for an unsupported configured backend."""

    def __init__(self, backend_name: str) -> None:
        self.backend_name = backend_name

    def analyze(self, request: VlmRequest) -> dict[str, Any]:
        return {
            "status": "skipped",
            "backend": self.backend_name,
            "message": f"Unsupported VLM backend: {self.backend_name}",
        }


class FallbackVlmEngine(VlmEngine):
    """Run a fallback adapter only when the primary adapter does not succeed."""

    def __init__(self, primary: VlmEngine, fallback: VlmEngine) -> None:
        self.primary = primary
        self.fallback = fallback

    def analyze(self, request: VlmRequest) -> dict[str, Any]:
        primary_result = self.primary.analyze(request)
        if primary_result.get("status") == "ok":
            return primary_result

        fallback_result = self.fallback.analyze(request)
        if fallback_result.get("status") == "ok":
            fallback_result.setdefault("primary_error", primary_result.get("error"))
            fallback_result.setdefault("primary_backend", primary_result.get("backend"))
            return fallback_result

        combined = dict(primary_result)
        combined["fallback"] = fallback_result
        return combined


class OptionalVlmEngine(VlmEngine):
    """Apply enablement, input validation, progress, and persistence policy."""

    def __init__(self, delegate: VlmEngine, *, backend_name: str) -> None:
        self.delegate = delegate
        self.backend_name = backend_name

    def analyze(self, request: VlmRequest) -> dict[str, Any]:
        out_json = request.result_dir / f"{request.run_id}_v14_7_vlm_raw_output.json"
        if not request.enabled:
            result = {
                "status": "disabled",
                "backend": self.backend_name,
                "message": "VLM evidence is disabled by configuration.",
            }
            save_json(out_json, result)
            return result

        if request.image_path is None or not request.image_path.exists():
            result = {
                "status": "skipped",
                "backend": self.backend_name,
                "message": "No source image path available for VLM evidence.",
            }
            save_json(out_json, result)
            return result

        _emit(
            request,
            "running",
            "Running optional PaddleOCR-VL / VLM evidence pass.",
            backend=self.backend_name,
        )
        result = dict(self.delegate.analyze(request))
        result.setdefault("backend", self.backend_name)
        result["image_path"] = str(request.image_path)
        save_json(out_json, result)
        _emit(
            request,
            result.get("status", "done"),
            "Optional VLM evidence pass finished.",
            backend=self.backend_name,
            vlm_status=result.get("status"),
            error=result.get("error"),
        )
        return result


__all__ = [
    "DisabledVlmEngine",
    "FallbackVlmEngine",
    "OptionalVlmEngine",
    "UnsupportedVlmEngine",
]
