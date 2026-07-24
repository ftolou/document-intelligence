"""Trusted deployment configuration for the standalone VLM HTTP service."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from receipt_intelligence.app_version import get_app_version
from receipt_intelligence.runtime.paths import RuntimePaths


@dataclass(frozen=True, slots=True)
class VlmHttpSettings:
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
    def from_environment(cls) -> "VlmHttpSettings":
        package_dir = Path(__file__).resolve().parent
        project_root = Path(os.getenv("APP_PROJECT_ROOT", Path.cwd())).resolve()
        candidate_root = package_dir.parents[3]
        if not (project_root / "VERSION").exists() and (candidate_root / "VERSION").exists():
            project_root = candidate_root

        runtime_paths = RuntimePaths.from_environment(project_root)
        upload_dir = Path(os.getenv("VLM_UPLOAD_DIR", runtime_paths.uploads_dir)).resolve()
        results_dir = Path(
            os.getenv("VLM_RESULTS_DIR", runtime_paths.var_root / "vlm_service")
        ).resolve()
        configured_roots = os.getenv("VLM_ALLOWED_INPUT_ROOTS", "").strip()
        roots = (
            tuple(
                Path(value).expanduser().resolve()
                for value in configured_roots.split(os.pathsep)
                if value
            )
            if configured_roots
            else (runtime_paths.var_root.resolve(), upload_dir)
        )
        roots = tuple(dict.fromkeys(roots))

        return cls(
            app_version=os.getenv("VLM_SERVICE_VERSION", get_app_version()),
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


__all__ = ["VlmHttpSettings"]
