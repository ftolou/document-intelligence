#!/usr/bin/env python3
"""Flask transport for the standalone visual-model service."""

from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path
from typing import Any

from flask import Flask, jsonify, request
from werkzeug.utils import secure_filename

from receipt_intelligence.application.vlm import VlmAnalysisService
from receipt_intelligence.composition import build_vlm_service_engine
from receipt_intelligence.entrypoints.vlm_http.settings import VlmHttpSettings


def is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def resolve_allowed_image_path(
    raw_path: object,
    allowed_roots: tuple[Path, ...],
) -> tuple[Path | None, str | None]:
    try:
        path = Path(str(raw_path)).expanduser().resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        return None, f"Image path cannot be resolved: {exc}"

    if not path.is_file():
        return None, f"Image path is not a file: {path}"
    if not any(is_within(path, root) for root in allowed_roots):
        return None, "Image path is outside the configured VLM input roots."
    return path, None


def resolve_image_from_request(
    settings: VlmHttpSettings,
) -> tuple[Path | None, str | None]:
    uploaded = request.files.get("file")
    if uploaded is not None:
        if not uploaded.filename:
            return None, "Uploaded file has no filename."
        name = secure_filename(uploaded.filename) or f"vlm_upload_{int(time.time())}.jpg"
        path = (settings.upload_dir / name).resolve()
        if not is_within(path, settings.upload_dir):
            return None, "Uploaded filename resolves outside the upload directory."
        uploaded.save(path)
        return path, None

    payload = request.get_json(silent=True) or {}
    image_path = payload.get("image_path")
    if not image_path:
        return None, "Missing image_path in JSON body or file upload field."
    return resolve_allowed_image_path(image_path, settings.allowed_input_roots)


def transformers_runtime_status() -> dict[str, Any]:
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


def gpu_runtime_status(device: str) -> dict[str, Any]:
    status: dict[str, Any] = {"requested_device": device}
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
        process = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.used", "--format=csv,noheader"],
            text=True,
            capture_output=True,
            timeout=5,
        )
        status["nvidia_smi_returncode"] = process.returncode
        status["nvidia_smi_stdout"] = process.stdout.strip()[-1000:]
        status["nvidia_smi_stderr"] = process.stderr.strip()[-1000:]
    except Exception as exc:
        status["nvidia_smi_error"] = f"{type(exc).__name__}: {exc}"
    return status


def create_app(
    *,
    settings: VlmHttpSettings | None = None,
    analysis_service: VlmAnalysisService | None = None,
) -> Flask:
    settings = settings or VlmHttpSettings.from_environment()
    settings.upload_dir.mkdir(parents=True, exist_ok=True)
    settings.results_dir.mkdir(parents=True, exist_ok=True)
    analysis_service = analysis_service or VlmAnalysisService(
        engine=build_vlm_service_engine(
            backend_name=settings.backend,
            runner_name=settings.runner,
            trusted_command=settings.command,
        ),
        results_dir=settings.results_dir,
        service_version=settings.app_version,
        timeout_seconds=settings.timeout_seconds,
        max_side_limit=settings.max_side_limit,
        runner_name=settings.runner,
        engine_name=settings.engine,
        device_name=settings.device,
    )

    flask_app = Flask(__name__)
    flask_app.config["MAX_CONTENT_LENGTH"] = settings.max_upload_mb * 1024 * 1024
    flask_app.extensions["vlm_settings"] = settings
    flask_app.extensions["vlm_analysis_service"] = analysis_service

    @flask_app.get("/health")
    def health():
        return jsonify(
            {
                "ok": True,
                "version": settings.app_version,
                "backend": settings.backend,
                "runner": settings.runner,
                "engine": settings.engine,
                "device": settings.device,
                "cpu_fallback_allowed": settings.allow_cpu_fallback,
                "timeout_seconds": settings.timeout_seconds,
                "max_side_limit": settings.max_side_limit,
                "has_command": bool(settings.command.strip()),
                "gpu": gpu_runtime_status(settings.device),
                "transformers_runtime": transformers_runtime_status(),
                "cache_paths": {
                    "paddleocr": "/root/.paddleocr",
                    "paddlex": os.getenv("PADDLEX_HOME", "/root/.paddlex"),
                    "huggingface": os.getenv("HF_HOME", "/root/.cache/huggingface"),
                    "torch": os.getenv("TORCH_HOME", "/root/.cache/torch"),
                    "paddle": os.getenv("PADDLE_HOME", "/root/.cache/paddle"),
                },
            }
        )

    @flask_app.post("/api/vlm/analyze")
    def analyze():
        started = time.perf_counter()
        payload = request.get_json(silent=True) or {}
        requested_run_id = str(payload.get("run_id") or f"vlm_{int(started * 1000)}")
        run_id = secure_filename(requested_run_id)[:120] or f"vlm_{int(started * 1000)}"
        image_path, error = resolve_image_from_request(settings)
        if error:
            return jsonify(
                {
                    "status": "error",
                    "backend": "receipt_vlm_service",
                    "error": error,
                    "duration_seconds": round(time.perf_counter() - started, 2),
                }
            ), 400

        assert image_path is not None
        result = analysis_service.execute(image_path=image_path, run_id=run_id)
        return jsonify(result)

    return flask_app


_settings = VlmHttpSettings.from_environment()
app = create_app(settings=_settings)


def main() -> None:
    app.run(host=_settings.host, port=_settings.port, debug=False)


if __name__ == "__main__":
    main()


__all__ = [
    "app",
    "create_app",
    "gpu_runtime_status",
    "is_within",
    "main",
    "resolve_allowed_image_path",
    "resolve_image_from_request",
    "transformers_runtime_status",
]
