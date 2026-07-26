"""Environment-backed configuration for the standalone VLM service."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

DEFAULT_VERSION = "1.0.0"


def _read_version(project_root: Path) -> str:
    configured = Path(os.getenv("APP_VERSION_FILE", project_root / "VERSION"))
    try:
        value = configured.read_text(encoding="utf-8").strip()
        return value or DEFAULT_VERSION
    except Exception:
        return DEFAULT_VERSION


def _path(key: str, default: Path) -> Path:
    value = str(os.getenv(key, "") or "").strip()
    return Path(value).expanduser().resolve() if value else default.resolve()


@dataclass(frozen=True, slots=True)
class VlmSettings:
    app_version: str
    upload_dir: Path
    results_dir: Path
    allowed_input_roots: tuple[Path, ...]
    backend: str
    command: str
    timeout_seconds: float
    max_upload_mb: int
    max_side_limit: int
    runner: str
    engine: str
    device: str
    allow_cpu_fallback: bool
    host: str
    port: int

    @classmethod
    def from_environment(cls) -> VlmSettings:
        project_root = Path(os.getenv("APP_PROJECT_ROOT", "/app")).resolve()
        var_root = _path("VAR_DIR", project_root / "var")
        upload_dir = _path("VLM_UPLOAD_DIR", var_root / "uploads")
        results_dir = _path("VLM_RESULTS_DIR", var_root / "vlm_service")

        configured_roots = os.getenv("VLM_ALLOWED_INPUT_ROOTS", "").strip()
        if configured_roots:
            roots = tuple(
                Path(value).expanduser().resolve()
                for value in configured_roots.split(os.pathsep)
                if value
            )
        else:
            roots = (var_root.resolve(), upload_dir.resolve())
        roots = tuple(dict.fromkeys(roots))

        return cls(
            app_version=os.getenv("VLM_SERVICE_VERSION", _read_version(project_root)),
            upload_dir=upload_dir,
            results_dir=results_dir,
            allowed_input_roots=roots,
            backend=os.getenv("VLM_SERVICE_BACKEND", "paddleocr_vl").strip(),
            command=os.getenv("VLM_SERVICE_COMMAND", ""),
            timeout_seconds=float(os.getenv("VLM_TIMEOUT_SECONDS", "900")),
            max_upload_mb=int(os.getenv("VLM_MAX_UPLOAD_MB", "25")),
            max_side_limit=int(os.getenv("VLM_SERVICE_MAX_SIDE_LIMIT", "1600")),
            runner=os.getenv("VLM_SERVICE_RUNNER", "cli").strip().lower(),
            engine=os.getenv("VLM_ENGINE", "transformers").strip(),
            device=os.getenv("VLM_DEVICE", "gpu:0").strip(),
            allow_cpu_fallback=os.getenv("VLM_ALLOW_CPU_FALLBACK", "0").strip().lower()
            in {"1", "true", "yes", "on"},
            host=os.getenv("VLM_SERVICE_HOST", "0.0.0.0"),
            port=int(os.getenv("VLM_SERVICE_PORT", "7870")),
        )


__all__ = ["VlmSettings"]
