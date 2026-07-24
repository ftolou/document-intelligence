"""Local PaddleOCR-VL Python API adapter."""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

from receipt_intelligence.application.ports.vlm import VlmEngine, VlmRequest
from receipt_intelligence.runtime.json_values import jsonable


def run_paddle_python(image_path: Path, timeout_seconds: float) -> dict[str, Any]:
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
            "raw_result": jsonable(output),
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


class PaddlePythonVlmEngine(VlmEngine):
    """Execute PaddleOCR-VL through its Python API."""

    def analyze(self, request: VlmRequest) -> dict[str, Any]:
        if request.image_path is None:
            return {
                "status": "skipped",
                "backend": "paddleocr_vl_python",
                "message": "No image path was provided.",
            }
        return run_paddle_python(request.image_path, request.timeout_seconds)


__all__ = ["PaddlePythonVlmEngine", "run_paddle_python"]
