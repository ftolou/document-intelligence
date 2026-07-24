"""HTTP client adapter for the standalone VLM service."""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from receipt_intelligence.application.ports.vlm import VlmEngine, VlmRequest


def call_remote_vlm(
    service_url: str, image_path: Path, run_id: str, timeout_seconds: float
) -> dict[str, Any]:
    """Call the separate V14.7 receipt-vlm service over HTTP.

    The main receipt-app container does not import PaddleOCR-VL. It sends only
    the shared image path and receives raw visual/document evidence JSON. The
    service is expected to mount the same uploads/outputs volumes at /app.
    """
    started = time.perf_counter()
    base = (service_url or "").strip().rstrip("/")
    if not base:
        return {
            "status": "error",
            "backend": "http_service",
            "error": "VLM_SERVICE_URL is empty.",
            "duration_seconds": round(time.perf_counter() - started, 2),
        }
    url = base if base.endswith("/api/vlm/analyze") else f"{base}/api/vlm/analyze"
    payload = {
        "image_path": str(image_path),
        "run_id": run_id,
        "mode": "document_visual_evidence",
    }
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_seconds) as resp:
            text = resp.read().decode("utf-8", errors="replace")
            data = json.loads(text) if text.strip() else {}
        if not isinstance(data, dict):
            data = {"raw_result": data}
        data.setdefault("status", "ok")
        data.setdefault("backend", "http_service")
        data["service_url"] = url
        data["duration_seconds"] = round(time.perf_counter() - started, 2)
        return data
    except urllib.error.HTTPError as exc:
        try:
            err_text = exc.read().decode("utf-8", errors="replace")
        except Exception:
            err_text = str(exc)
        return {
            "status": "error",
            "backend": "http_service",
            "service_url": url,
            "error": f"HTTP {exc.code}: {err_text[-2000:]}",
            "duration_seconds": round(time.perf_counter() - started, 2),
        }
    except Exception as exc:
        return {
            "status": "error",
            "backend": "http_service",
            "service_url": url,
            "error": f"{type(exc).__name__}: {exc}",
            "duration_seconds": round(time.perf_counter() - started, 2),
        }


class RemoteVlmClient(VlmEngine):
    """Call a configured standalone VLM HTTP service."""

    def __init__(self, service_url: str) -> None:
        self.service_url = service_url

    def analyze(self, request: VlmRequest) -> dict[str, Any]:
        if request.image_path is None:
            return {
                "status": "skipped",
                "backend": "http_service",
                "message": "No image path was provided.",
            }
        return call_remote_vlm(
            self.service_url,
            request.image_path,
            request.run_id,
            request.timeout_seconds,
        )


__all__ = ["RemoteVlmClient", "call_remote_vlm"]
