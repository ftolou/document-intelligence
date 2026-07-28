"""Application policies that compose the mandatory visual-model adapter."""

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


class UnsupportedVlmEngine(VlmEngine):
    """Return a structured result for an unsupported configured backend."""

    def __init__(self, backend_name: str) -> None:
        self.backend_name = backend_name

    def analyze(self, request: VlmRequest) -> dict[str, Any]:
        return {
            "status": "error",
            "backend": self.backend_name,
            "error": f"Unsupported VLM backend: {self.backend_name}",
        }


class RequiredVlmEngine(VlmEngine):
    """Apply input validation, progress, and persistence for mandatory VLM execution."""

    def __init__(self, delegate: VlmEngine, *, backend_name: str) -> None:
        self.delegate = delegate
        self.backend_name = backend_name

    def analyze(self, request: VlmRequest) -> dict[str, Any]:
        out_json = request.result_dir / f"{request.run_id}_v14_7_vlm_raw_output.json"
        if request.image_path is None or not request.image_path.exists():
            raise FileNotFoundError("A source receipt image is required for PaddleOCR-VL.")

        _emit(
            request,
            "running",
            "Running mandatory PaddleOCR-VL evidence extraction.",
            backend=self.backend_name,
        )
        result = dict(self.delegate.analyze(request))
        result.setdefault("backend", self.backend_name)
        result["image_path"] = str(request.image_path)
        save_json(out_json, result)
        _emit(
            request,
            result.get("status", "done"),
            "PaddleOCR-VL evidence extraction finished.",
            backend=self.backend_name,
            vlm_status=result.get("status"),
            error=result.get("error"),
        )
        return result


__all__ = [
    "RequiredVlmEngine",
    "UnsupportedVlmEngine",
]
