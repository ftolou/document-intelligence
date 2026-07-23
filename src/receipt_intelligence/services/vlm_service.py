#!/usr/bin/env python3
"""Standalone PaddleOCR-VL HTTP service.

This service intentionally runs in a separate container from the main receipt app.
It owns the heavier PaddleOCR-VL / doc-parser dependencies and exposes a small
HTTP API. The main app sends a shared image path and receives raw visual evidence.

Endpoints:
  GET  /health
  POST /api/vlm/analyze

Request JSON:
  {"image_path": "/app/var/jobs/<job_id>/...jpg", "run_id": "abc123"}

Response JSON:
  {"status": "ok|error|unavailable", "backend": "...", "raw_result": ...}
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

try:
    from PIL import Image
except Exception:  # pragma: no cover - service still reports a clear error if resizing is requested
    Image = None  # type: ignore
from typing import Any

from flask import Flask, jsonify, request
from werkzeug.utils import secure_filename

# Reuse the best-effort PaddleOCR-VL adapters from the main codebase. They only
# import PaddleOCR-VL inside the function, so importing this module is cheap.
from receipt_intelligence.app_version import get_app_version
from receipt_intelligence.engines.vl_engine import (
    _jsonable,
    _run_command_backend,
    _run_paddleocr_vl_cli,
    _run_paddleocr_vl_python,
)
from receipt_intelligence.runtime.paths import RuntimePaths

APP_VERSION = os.getenv("VLM_SERVICE_VERSION", get_app_version())
PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(os.getenv("APP_PROJECT_ROOT", Path.cwd())).resolve()
if not (PROJECT_ROOT / "VERSION").exists() and (PACKAGE_DIR.parents[2] / "VERSION").exists():
    PROJECT_ROOT = PACKAGE_DIR.parents[2]
BASE_DIR = PROJECT_ROOT
RUNTIME_PATHS = RuntimePaths.from_environment(BASE_DIR)
UPLOAD_DIR = Path(os.getenv("VLM_UPLOAD_DIR", RUNTIME_PATHS.uploads_dir))
RESULTS_DIR = Path(os.getenv("VLM_RESULTS_DIR", RUNTIME_PATHS.var_root / "vlm_service"))
BACKEND = os.getenv("VLM_SERVICE_BACKEND", "paddleocr_vl")
COMMAND = os.getenv("VLM_SERVICE_COMMAND", "")
TIMEOUT_SECONDS = float(os.getenv("VLM_TIMEOUT_SECONDS", "900"))
MAX_UPLOAD_MB = int(os.getenv("VLM_MAX_UPLOAD_MB", "25"))
MAX_SIDE_LIMIT = int(os.getenv("VLM_SERVICE_MAX_SIDE_LIMIT", "1600"))
RUNNER = os.getenv("VLM_SERVICE_RUNNER", "cli").strip().lower()
ENGINE = os.getenv("VLM_ENGINE", "transformers").strip()
DEVICE = os.getenv("VLM_DEVICE", "gpu:0").strip()
ALLOW_CPU_FALLBACK = os.getenv("VLM_ALLOW_CPU_FALLBACK", "0").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
VLM_DEVICE = DEVICE

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024


def _save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _resolve_image_from_request() -> tuple[Path | None, str | None]:
    """Return image path or error message.

    Path mode is preferred because docker-compose mounts the same canonical
    ``var/`` runtime root into both containers. File-upload mode is provided
    for manual testing.
    """
    if request.files.get("file") is not None:
        f = request.files["file"]
        if not f.filename:
            return None, "Uploaded file has no filename."
        name = secure_filename(f.filename) or f"vlm_upload_{int(time.time())}.jpg"
        path = UPLOAD_DIR / name
        f.save(path)
        return path, None

    payload = request.get_json(silent=True) or {}
    image_path = payload.get("image_path") or payload.get("path")
    if not image_path:
        return None, "Missing image_path in JSON body or file upload field."
    path = Path(str(image_path))
    if not path.exists():
        return None, f"Image path does not exist inside receipt-vlm container: {path}"
    if not path.is_file():
        return None, f"Image path is not a file: {path}"
    return path, None


def _transformers_runtime_status() -> dict[str, Any]:
    status: dict[str, Any] = {}
    try:
        import torch  # type: ignore

        status["torch_version"] = getattr(torch, "__version__", None)
        status["torch_cuda_available"] = bool(torch.cuda.is_available())
        status["torch_cuda_device_count"] = (
            int(torch.cuda.device_count()) if torch.cuda.is_available() else 0
        )
        if torch.cuda.is_available():
            status["torch_gpu_name"] = torch.cuda.get_device_name(0)
    except Exception as exc:
        status["torch_error"] = f"{type(exc).__name__}: {exc}"
    try:
        import torchvision  # type: ignore

        status["torchvision_version"] = getattr(torchvision, "__version__", None)
    except Exception as exc:
        status["torchvision_error"] = f"{type(exc).__name__}: {exc}"
    try:
        import docx  # noqa: F401

        status["python_docx"] = True
    except Exception as exc:
        status["python_docx_error"] = f"{type(exc).__name__}: {exc}"
    try:
        from transformers import AutoImageProcessor, AutoModelForObjectDetection  # noqa: F401

        status["transformers_image_imports"] = True
    except Exception as exc:
        status["transformers_image_imports_error"] = f"{type(exc).__name__}: {exc}"
    return status


def _gpu_runtime_status() -> dict[str, Any]:
    status: dict[str, Any] = {"requested_device": VLM_DEVICE}
    try:
        import paddle  # type: ignore

        status["paddle_version"] = getattr(paddle, "__version__", None)
        status["is_compiled_with_cuda"] = bool(
            getattr(paddle.device, "is_compiled_with_cuda", lambda: False)()
        )
        try:
            status["cuda_device_count"] = int(
                getattr(paddle.device.cuda, "device_count", lambda: 0)()
            )
        except Exception as exc:
            status["cuda_device_count_error"] = f"{type(exc).__name__}: {exc}"
    except Exception as exc:
        status["paddle_import_error"] = f"{type(exc).__name__}: {exc}"
    try:
        import subprocess

        proc = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.used", "--format=csv,noheader"],
            text=True,
            capture_output=True,
            timeout=5,
        )
        status["nvidia_smi_returncode"] = proc.returncode
        status["nvidia_smi_stdout"] = proc.stdout.strip()[-1000:]
        status["nvidia_smi_stderr"] = proc.stderr.strip()[-1000:]
    except Exception as exc:
        status["nvidia_smi_error"] = f"{type(exc).__name__}: {exc}"
    return status


@app.get("/health")
def health():
    return jsonify(
        {
            "ok": True,
            "version": APP_VERSION,
            "backend": BACKEND,
            "runner": RUNNER,
            "engine": ENGINE,
            "device": DEVICE,
            "cpu_fallback_allowed": ALLOW_CPU_FALLBACK,
            "timeout_seconds": TIMEOUT_SECONDS,
            "max_side_limit": MAX_SIDE_LIMIT,
            "has_command": bool(COMMAND.strip()),
            "gpu": _gpu_runtime_status(),
            "transformers_runtime": _transformers_runtime_status(),
            "cache_paths": {
                "paddleocr": "/root/.paddleocr",
                "paddlex": os.getenv("PADDLEX_HOME", "/root/.paddlex"),
                "huggingface": os.getenv("HF_HOME", "/root/.cache/huggingface"),
                "torch": os.getenv("TORCH_HOME", "/root/.cache/torch"),
                "paddle": os.getenv("PADDLE_HOME", "/root/.cache/paddle"),
            },
        }
    )


def _prepare_image_for_vlm(
    image_path: Path, result_dir: Path, max_side: int
) -> tuple[Path, dict[str, Any]]:
    """Create a resized copy for VLM if the receipt image is very large.

    PaddleOCR-VL/doc-parser can be very slow on CPU for full-resolution phone
    receipt photos. The main OCR layer still uses the full prepared image. The
    VLM service uses this bounded copy only as additional evidence.
    """
    meta: dict[str, Any] = {"original_path": str(image_path), "max_side_limit": max_side}
    if max_side <= 0:
        meta["resized"] = False
        meta["reason"] = "disabled"
        return image_path, meta
    if Image is None:
        meta["resized"] = False
        meta["reason"] = "pillow_unavailable"
        return image_path, meta
    try:
        with Image.open(image_path) as im:
            im = im.convert("RGB")
            w, h = im.size
            meta["original_width"] = w
            meta["original_height"] = h
            side = max(w, h)
            if side <= max_side:
                meta["resized"] = False
                meta["prepared_width"] = w
                meta["prepared_height"] = h
                return image_path, meta
            scale = max_side / float(side)
            nw = max(1, int(round(w * scale)))
            nh = max(1, int(round(h * scale)))
            out_path = result_dir / f"{image_path.stem}_vlm_resized_{max_side}.jpg"
            im = im.resize((nw, nh))
            im.save(out_path, quality=92, optimize=True)
            meta.update(
                {
                    "resized": True,
                    "prepared_path": str(out_path),
                    "prepared_width": nw,
                    "prepared_height": nh,
                    "scale": scale,
                }
            )
            return out_path, meta
    except Exception as exc:
        meta["resized"] = False
        meta["error"] = f"{type(exc).__name__}: {exc}"
        return image_path, meta


@app.post("/api/vlm/analyze")
def analyze():
    started = time.perf_counter()
    payload = request.get_json(silent=True) or {}
    run_id = str(payload.get("run_id") or f"vlm_{int(started * 1000)}")
    image_path, err = _resolve_image_from_request()
    if err:
        return jsonify(
            {
                "status": "error",
                "backend": "receipt_vlm_service",
                "error": err,
                "duration_seconds": round(time.perf_counter() - started, 2),
            }
        ), 400

    assert image_path is not None
    result_dir = RESULTS_DIR / run_id
    result_dir.mkdir(parents=True, exist_ok=True)
    out_json = result_dir / f"{run_id}_vlm_service_raw.json"

    backend = str(payload.get("backend") or BACKEND or "paddleocr_vl").lower()
    command = str(payload.get("command") or COMMAND or "")
    timeout = float(payload.get("timeout_seconds") or TIMEOUT_SECONDS)
    max_side = int(payload.get("max_side_limit") or MAX_SIDE_LIMIT)
    runner = str(payload.get("runner") or RUNNER or "auto").strip().lower()

    prepared_image_path, prepare_meta = _prepare_image_for_vlm(image_path, result_dir, max_side)

    if command.strip():
        result = _run_command_backend(command, prepared_image_path, out_json, timeout)
        result["backend"] = "receipt_vlm_service_command"
    elif backend in {"paddleocr_vl", "paddleocr-vl", "paddleocrvl"}:
        result = {
            "status": "skipped",
            "backend": "receipt_vlm_service",
            "message": "No runner executed.",
        }
        if runner in {"auto", "python", "python_api"}:
            result = _run_paddleocr_vl_python(prepared_image_path, timeout)
        if (runner in {"auto", "cli", "doc_parser"}) and result.get("status") != "ok":
            cli_result = _run_paddleocr_vl_cli(prepared_image_path, result_dir, timeout)
            if cli_result.get("status") == "ok":
                cli_result["python_api_error"] = result.get("error")
                result = cli_result
            else:
                result["cli_fallback"] = cli_result
        result["backend"] = f"receipt_vlm_service:{result.get('backend')}"
    else:
        result = {
            "status": "skipped",
            "backend": "receipt_vlm_service",
            "message": f"Unsupported VLM service backend: {backend}",
        }

    result["service_version"] = APP_VERSION
    result["image_path"] = str(image_path)
    result["prepared_image_path"] = str(prepared_image_path)
    result["image_prepare"] = prepare_meta
    result["runner"] = runner
    result["engine"] = ENGINE
    result["device"] = VLM_DEVICE
    result["run_id"] = run_id
    result["duration_seconds"] = result.get(
        "duration_seconds", round(time.perf_counter() - started, 2)
    )
    result = _jsonable(result)
    _save_json(out_json, result)
    return jsonify(result)


def main() -> None:
    host = os.getenv("VLM_SERVICE_HOST", "0.0.0.0")
    port = int(os.getenv("VLM_SERVICE_PORT", "7870"))
    app.run(host=host, port=port, debug=False)


if __name__ == "__main__":
    main()
