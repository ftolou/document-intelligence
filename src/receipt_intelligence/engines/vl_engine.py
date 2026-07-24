#!/usr/bin/env python3
"""Optional visual-document evidence engine.

This module intentionally does not participate in the normal fast path.
It is used only when enabled and validation reports that the first LLM parse
is not safe enough. The output is evidence, not final receipt semantics.

Supported modes:
  - backend="http_service": call a separate receipt-vlm HTTP service container.
  - backend="paddleocr_vl": legacy/local adapter, disabled by default in receipt-app.
  - command: server-configured command template with {image} and {output_json}.

The command mode is deliberately generic because PaddleOCR-VL CLI/API details can
change between PaddleOCR releases. Commands are parsed into argument vectors and
run without a shell. The app stays runnable without VLM installed.
"""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable

from receipt_intelligence.runtime.command_execution import expand_command_template


def _save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _emit(
    callback: Callable[[dict[str, Any]], None] | None,
    stage: str,
    status: str,
    message: str,
    **details: Any,
) -> None:
    if callback is None:
        return
    try:
        callback(
            {
                "stage": stage,
                "status": status,
                "message": message,
                "details": details,
                "source": "vl_engine",
            }
        )
    except Exception:
        pass


def _jsonable(obj: Any) -> Any:
    """Best-effort conversion of Paddle/Pydantic/custom results to JSONable data."""
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_jsonable(x) for x in obj]
    for attr in ("json", "to_dict", "dict", "model_dump"):
        if hasattr(obj, attr):
            try:
                val = getattr(obj, attr)
                val = val() if callable(val) else val
                if isinstance(val, str):
                    try:
                        return json.loads(val)
                    except Exception:
                        return val
                return _jsonable(val)
            except Exception:
                pass
    if hasattr(obj, "res"):
        try:
            return _jsonable(getattr(obj, "res"))
        except Exception:
            pass
    return str(obj)


def _run_paddleocr_vl_python(image_path: Path, timeout_seconds: float) -> dict[str, Any]:
    """Run PaddleOCR-VL via Python API when available.

    Hardened for PaddleOCR-VL / PaddlePaddle CUDA initialization:
      - set CUDA/Paddle environment before importing paddleocr
      - explicitly set Paddle's current device
      - force the expected Place to CUDAPlace(0) / CPUPlace when Paddle exposes it
      - guard Paddle's BF16 capability helper against Place(undefined:0)
      - pass engine="transformers" first, matching the known-good CLI path
    """
    started = time.perf_counter()
    requested_device = os.getenv("VLM_DEVICE", "gpu:0").strip().lower() or "gpu:0"
    engine = os.getenv("VLM_ENGINE", "transformers").strip() or "transformers"
    precision = os.getenv("VLM_PRECISION", "fp16").strip().lower() or "fp16"

    wants_gpu = requested_device.startswith(("gpu", "cuda"))
    if wants_gpu:
        # These must exist before paddleocr imports its Paddle/PaddleX stack.
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
        os.environ.setdefault("FLAGS_selected_gpus", "0")
        os.environ["PADDLE_DEVICE"] = "gpu:0"
    else:
        os.environ["PADDLE_DEVICE"] = "cpu"

    runtime: dict[str, Any] = {
        "requested_device": requested_device,
        "engine": engine,
        "precision": precision,
        "env_cuda_visible_devices": os.getenv("CUDA_VISIBLE_DEVICES"),
        "env_flags_selected_gpus": os.getenv("FLAGS_selected_gpus"),
        "env_paddle_device": os.getenv("PADDLE_DEVICE"),
    }

    try:
        import paddle  # type: ignore
    except Exception as exc:
        return {
            "status": "unavailable",
            "backend": "paddleocr_vl_python",
            "device": requested_device,
            "engine": engine,
            "precision": precision,
            "runtime": runtime,
            "error": f"Paddle import failed: {type(exc).__name__}: {exc}",
            "duration_seconds": round(time.perf_counter() - started, 2),
        }

    def _paddle_place_for_device(resolved: str) -> Any:
        if resolved.startswith("gpu"):
            return paddle.CUDAPlace(0)
        return paddle.CPUPlace()

    def _force_expected_place(place: Any) -> list[str]:
        notes: list[str] = []
        # Paddle exposes this helper in different modules depending on version.
        candidates = []
        try:
            candidates.append(paddle.framework._set_expected_place)  # type: ignore[attr-defined]
        except Exception:
            pass
        try:
            candidates.append(paddle.base.framework._set_expected_place)  # type: ignore[attr-defined]
        except Exception:
            pass
        try:
            candidates.append(paddle.fluid.framework._set_expected_place)  # type: ignore[attr-defined]
        except Exception:
            pass
        for fn in candidates:
            try:
                fn(place)
                notes.append(
                    f"set_expected_place_ok:{getattr(fn, '__module__', '?')}.{getattr(fn, '__name__', '?')}"
                )
            except Exception as exc:
                notes.append(f"set_expected_place_failed:{type(exc).__name__}:{exc}")
        return notes

    def _install_bf16_guard(place: Any) -> str:
        """Avoid PaddleOCR-VL crashing when Paddle passes Place(undefined:0).

        Some Paddle/PaddleX combinations call paddle.amp.is_bfloat16_supported()
        with the current expected place. In this Docker/Python API path that place
        can become Place(undefined:0), even after paddle.set_device('gpu:0').
        For VLM inference we prefer a conservative False over crashing.
        """
        try:
            original = paddle.amp.is_bfloat16_supported
        except Exception as exc:
            return f"bf16_guard_unavailable:{type(exc).__name__}:{exc}"

        def guarded_is_bfloat16_supported(arg: Any = None) -> bool:  # type: ignore[override]
            try:
                return bool(original(place if arg is None else arg))
            except TypeError as exc:
                text = str(exc)
                if "Place(undefined" in text or "incompatible function arguments" in text:
                    try:
                        return bool(original(place))
                    except Exception:
                        return False
                raise
            except Exception:
                return False

        try:
            paddle.amp.is_bfloat16_supported = guarded_is_bfloat16_supported  # type: ignore[assignment]
            return "bf16_guard_installed"
        except Exception as exc:
            return f"bf16_guard_failed:{type(exc).__name__}:{exc}"

    try:
        cuda_available = bool(paddle.device.is_compiled_with_cuda())
        try:
            cuda_count = int(paddle.device.cuda.device_count()) if cuda_available else 0
        except Exception:
            cuda_count = 0

        if wants_gpu:
            if not cuda_available or cuda_count < 1:
                raise RuntimeError(
                    f"GPU requested ({requested_device}) but Paddle CUDA is unavailable "
                    f"(is_compiled_with_cuda={cuda_available}, cuda_device_count={cuda_count})."
                )
            resolved_device = "gpu:0"
        else:
            resolved_device = "cpu"

        paddle.set_device(resolved_device)
        place = _paddle_place_for_device(resolved_device)
        place_notes = _force_expected_place(place)
        bf16_guard_status = _install_bf16_guard(place)

        runtime.update(
            {
                "is_compiled_with_cuda": cuda_available,
                "cuda_device_count": cuda_count,
                "resolved_device": resolved_device,
                "current_device_after_set": paddle.get_device(),
                "forced_place": str(place),
                "expected_place_notes": place_notes[-5:],
                "bf16_guard": bf16_guard_status,
            }
        )

        try:
            runtime["bf16_supported_explicit_place"] = bool(paddle.amp.is_bfloat16_supported(place))
        except Exception as exc:
            runtime["bf16_supported_explicit_place_error"] = f"{type(exc).__name__}: {exc}"

        # Import PaddleOCRVL only after the Paddle device and BF16 guard are ready.
        try:
            from paddleocr import PaddleOCRVL  # type: ignore
        except Exception as exc:
            return {
                "status": "unavailable",
                "backend": "paddleocr_vl_python",
                "device": resolved_device,
                "engine": engine,
                "precision": precision,
                "runtime": runtime,
                "error": f"PaddleOCRVL import failed: {type(exc).__name__}: {exc}",
                "duration_seconds": round(time.perf_counter() - started, 2),
            }

        # Prefer no explicit device in PaddleOCRVL once Paddle's global device is set,
        # because some versions parse device='gpu:0' inconsistently inside PaddleX.
        # Keep engine='transformers' first to match the working CLI behavior.
        init_attempts: list[dict[str, Any]] = [
            {"engine": engine, "precision": precision},
            {"engine": engine},
            {
                "device": "gpu" if resolved_device.startswith("gpu") else "cpu",
                "engine": engine,
                "precision": precision,
            },
            {"device": "gpu" if resolved_device.startswith("gpu") else "cpu", "engine": engine},
            {"device": resolved_device, "engine": engine},
            {"device": resolved_device},
            {},
        ]
        init_errors: list[str] = []
        pipeline = None
        for kwargs in init_attempts:
            try:
                runtime["init_kwargs_try"] = kwargs
                pipeline = PaddleOCRVL(**kwargs)
                runtime["init_kwargs_used"] = kwargs
                break
            except TypeError as exc:
                init_errors.append(f"{kwargs}: {type(exc).__name__}: {exc}")
                continue
            except Exception as exc:
                init_errors.append(f"{kwargs}: {type(exc).__name__}: {exc}")
                continue

        if pipeline is None:
            raise RuntimeError(
                "All PaddleOCRVL initialization attempts failed: " + " || ".join(init_errors[-4:])
            )

        # Re-assert device after PaddleX construction; some internals mutate context.
        paddle.set_device(resolved_device)
        _force_expected_place(place)
        runtime["current_device_before_predict"] = paddle.get_device()

        if hasattr(pipeline, "predict"):
            try:
                output = pipeline.predict(input=str(image_path))
            except TypeError:
                output = pipeline.predict(str(image_path))
        elif callable(pipeline):
            output = pipeline(str(image_path))
        else:
            raise RuntimeError("PaddleOCRVL object has neither predict() nor __call__().")

        return {
            "status": "ok",
            "backend": "paddleocr_vl_python",
            "device": resolved_device,
            "engine": engine,
            "precision": precision,
            "runtime": runtime,
            "init_errors": init_errors[-8:],
            "raw_result": _jsonable(output),
            "duration_seconds": round(time.perf_counter() - started, 2),
        }

    except Exception as exc:
        import traceback

        try:
            runtime["current_device_on_error"] = paddle.get_device()
        except Exception:
            pass

        return {
            "status": "error",
            "backend": "paddleocr_vl_python",
            "device": requested_device,
            "engine": engine,
            "precision": precision,
            "runtime": runtime,
            "error": f"PaddleOCRVL run failed: {type(exc).__name__}: {exc}",
            "traceback_tail": traceback.format_exc()[-12000:],
            "duration_seconds": round(time.perf_counter() - started, 2),
        }


def _run_paddleocr_vl_cli(
    image_path: Path, result_dir: Path, timeout_seconds: float
) -> dict[str, Any]:
    """Run PaddleOCR-VL via the doc_parser CLI.

    V14.7.9 locks the service to the standalone-tested route on this machine:
      paddleocr doc_parser -i IMAGE --save_path OUT --device gpu:0 --engine transformers

    The CLI writes JSON/MD/DOCX/images; this function collects JSON/MD/TXT/HTML
    content and also records all produced file paths for diagnostics.
    """
    started = time.perf_counter()
    out_dir = result_dir / "paddleocr_vl_cli_output"
    out_dir.mkdir(parents=True, exist_ok=True)
    device = os.getenv("VLM_DEVICE", "gpu:0").strip() or "gpu:0"
    engine = os.getenv("VLM_ENGINE", "transformers").strip()
    extra_args = os.getenv("VLM_CLI_EXTRA_ARGS", "").strip()

    cmd = [
        "paddleocr",
        "doc_parser",
        "-i",
        str(image_path),
        "--save_path",
        str(out_dir),
        "--device",
        device,
    ]
    if engine:
        cmd.extend(["--engine", engine])
    if extra_args:
        cmd.extend(shlex.split(extra_args))

    try:
        env = os.environ.copy()
        env.setdefault("CUDA_VISIBLE_DEVICES", "0")
        env.setdefault("FLAGS_selected_gpus", "0")
        env.setdefault("PADDLE_DEVICE", device)
        proc = subprocess.run(cmd, text=True, capture_output=True, timeout=timeout_seconds, env=env)
        files: list[dict[str, Any]] = []
        produced_files: list[str] = []
        for fp in sorted(out_dir.rglob("*")):
            if not fp.is_file():
                continue
            produced_files.append(str(fp))
            if fp.suffix.lower() not in {".json", ".md", ".txt", ".html"}:
                continue
            try:
                text = fp.read_text(encoding="utf-8", errors="replace")
            except Exception:
                continue
            content: Any = text
            if fp.suffix.lower() == ".json":
                try:
                    content = json.loads(text)
                except Exception:
                    pass
            files.append(
                {"path": str(fp), "name": fp.name, "suffix": fp.suffix.lower(), "content": content}
            )

        status = "ok" if proc.returncode == 0 and (files or produced_files) else "error"
        error = None
        if status != "ok":
            reason = []
            if proc.returncode != 0:
                reason.append(f"CLI exited with return code {proc.returncode}")
            if not files and not produced_files:
                reason.append("CLI produced no output files")
            if proc.stderr.strip():
                reason.append(proc.stderr.strip()[-1500:])
            elif proc.stdout.strip():
                reason.append(proc.stdout.strip()[-1500:])
            error = " | ".join(reason) or "PaddleOCR-VL CLI failed without stderr."

        return {
            "status": status,
            "backend": "paddleocr_vl_cli",
            "device": device,
            "engine": engine,
            "command": " ".join(shlex.quote(x) for x in cmd),
            "returncode": proc.returncode,
            "error": error,
            "stdout_tail": proc.stdout[-6000:],
            "stderr_tail": proc.stderr[-6000:],
            "output_dir": str(out_dir),
            "produced_files": produced_files,
            "raw_result": {"files": files},
            "duration_seconds": round(time.perf_counter() - started, 2),
        }
    except Exception as exc:
        return {
            "status": "error",
            "backend": "paddleocr_vl_cli",
            "device": device,
            "engine": engine,
            "command": " ".join(shlex.quote(x) for x in cmd),
            "error": f"{type(exc).__name__}: {exc}",
            "duration_seconds": round(time.perf_counter() - started, 2),
        }


def _run_command_backend(
    command_template: str, image_path: Path, output_json_path: Path, timeout_seconds: float
) -> dict[str, Any]:
    started = time.perf_counter()
    cmd: list[str] = []
    try:
        cmd = expand_command_template(
            command_template,
            {
                "image": str(image_path),
                "output_json": str(output_json_path),
                "output": str(output_json_path),
            },
        )
        proc = subprocess.run(
            cmd,
            shell=False,
            text=True,
            capture_output=True,
            timeout=timeout_seconds,
            cwd=str(image_path.parent),
        )
        raw_result: Any = None
        if output_json_path.exists():
            try:
                raw_result = json.loads(output_json_path.read_text(encoding="utf-8-sig"))
            except Exception:
                raw_result = output_json_path.read_text(encoding="utf-8", errors="replace")
        elif proc.stdout.strip():
            try:
                raw_result = json.loads(proc.stdout)
            except Exception:
                raw_result = proc.stdout
        return {
            "status": "ok" if proc.returncode == 0 else "error",
            "backend": "command",
            "command": shlex.join(cmd),
            "argv": cmd,
            "returncode": proc.returncode,
            "stdout_tail": proc.stdout[-4000:],
            "stderr_tail": proc.stderr[-4000:],
            "raw_result": _jsonable(raw_result),
            "duration_seconds": round(time.perf_counter() - started, 2),
        }
    except Exception as exc:
        return {
            "status": "error",
            "backend": "command",
            "command": shlex.join(cmd) if cmd else command_template,
            "argv": cmd,
            "error": f"{type(exc).__name__}: {exc}",
            "duration_seconds": round(time.perf_counter() - started, 2),
        }


def _run_http_service_backend(
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


def run_optional_vlm(
    *,
    image_path: Path | None,
    result_dir: Path,
    run_id: str,
    enabled: bool = False,
    backend: str = "http_service",
    service_url: str = "http://receipt-vlm:7870",
    command: str = "",
    timeout_seconds: float = 180.0,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Run optional VLM document parser and save raw output.

    Returns a JSONable object with status ok/skipped/unavailable/error.
    """
    out_json = result_dir / f"{run_id}_v14_7_vlm_raw_output.json"
    if not enabled:
        result = {
            "status": "disabled",
            "backend": backend,
            "message": "VLM evidence is disabled by configuration.",
        }
        _save_json(out_json, result)
        return result
    if image_path is None or not Path(image_path).exists():
        result = {
            "status": "skipped",
            "backend": backend,
            "message": "No source image path available for VLM evidence.",
        }
        _save_json(out_json, result)
        return result

    requested_backend = (backend or "http_service").strip()
    local_names = {"paddleocr_vl", "paddleocr-vl", "paddleocrvl", "local"}
    allow_local = os.getenv("VLM_ALLOW_LOCAL_BACKEND", "0").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    if requested_backend.lower() in local_names and not allow_local:
        _emit(
            progress_callback,
            "visual_evidence",
            "running",
            "Local PaddleOCR-VL backend requested, but V14.7 routes VLM through the separate receipt-vlm service.",
            requested_backend=requested_backend,
            routed_backend="http_service",
            service_url=service_url,
        )
        backend = "http_service"
    else:
        backend = requested_backend

    _emit(
        progress_callback,
        "visual_evidence",
        "running",
        "Running optional PaddleOCR-VL / VLM evidence pass.",
        backend=backend,
    )
    image_path = Path(image_path)
    if command.strip():
        result = _run_command_backend(command, image_path, out_json, timeout_seconds)
    elif backend.lower() in {"http_service", "http-service", "service", "vlm_service"}:
        result = _run_http_service_backend(service_url, image_path, run_id, timeout_seconds)
    elif backend.lower() in {"paddleocr_vl", "paddleocr-vl", "paddleocrvl"}:
        runner = os.getenv("VLM_SERVICE_RUNNER", os.getenv("VLM_RUNNER", "cli")).strip().lower()
        if runner in {"cli", "doc_parser", "paddleocr_cli"}:
            result = _run_paddleocr_vl_cli(image_path, result_dir, timeout_seconds)
        else:
            result = _run_paddleocr_vl_python(image_path, timeout_seconds)
            if result.get("status") != "ok":
                # Fallback to the official PaddleOCR doc_parser CLI when the Python
                # API exists but pipeline creation fails or the class is unavailable.
                cli_result = _run_paddleocr_vl_cli(image_path, result_dir, timeout_seconds)
                if cli_result.get("status") == "ok":
                    cli_result["python_api_error"] = result.get("error")
                    result = cli_result
                else:
                    result["cli_fallback"] = cli_result
    else:
        result = {
            "status": "skipped",
            "backend": backend,
            "message": f"Unsupported VLM backend: {backend}",
        }

    result["image_path"] = str(image_path)
    _save_json(out_json, result)
    _emit(
        progress_callback,
        "visual_evidence",
        result.get("status", "done"),
        "Optional VLM evidence pass finished.",
        backend=backend,
        vlm_status=result.get("status"),
        error=result.get("error"),
    )
    return result
