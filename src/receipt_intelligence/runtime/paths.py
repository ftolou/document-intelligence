"""Central runtime-directory resolution for the canonical ``var/`` layout."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path


def _path_from_env(
    environ: Mapping[str, str],
    key: str,
    default: Path,
) -> Path:
    value = str(environ.get(key, "") or "").strip()
    return Path(value).expanduser() if value else default


@dataclass(frozen=True)
class RuntimePaths:
    """Resolved runtime paths.

    Phase 6 completes the cutover to the canonical ``var/`` directory. Legacy
    roots such as ``outputs/results`` and ``data`` are no longer searched or
    used as implicit fallbacks.
    """

    project_root: Path
    var_root: Path

    uploads_dir: Path
    jobs_dir: Path
    database_dir: Path
    reports_dir: Path
    batch_input_dir: Path
    logs_dir: Path
    receipt_db_path: Path

    @classmethod
    def from_environment(
        cls,
        project_root: Path,
        environ: Mapping[str, str] | None = None,
    ) -> RuntimePaths:
        env = os.environ if environ is None else environ
        root = Path(project_root).resolve()

        requested_layout = str(env.get("RUNTIME_LAYOUT", "var") or "var").strip().lower()
        if requested_layout != "var":
            raise ValueError(
                "Only the canonical var runtime layout is supported. "
                "Migrate legacy runtime data before starting v1.21.0."
            )

        var_root = _path_from_env(env, "VAR_DIR", root / "var").resolve()
        uploads_dir = _path_from_env(env, "UPLOAD_DIR", var_root / "uploads").resolve()
        jobs_dir = _path_from_env(env, "RESULTS_DIR", var_root / "jobs").resolve()
        database_dir = _path_from_env(env, "DATA_DIR", var_root / "database").resolve()
        reports_dir = _path_from_env(env, "REPORTS_DIR", var_root / "reports").resolve()
        batch_input_dir = _path_from_env(
            env,
            "BATCH_INPUT_DIR",
            var_root / "batch_input",
        ).resolve()
        logs_dir = _path_from_env(env, "LOGS_DIR", var_root / "logs").resolve()
        receipt_db_path = _path_from_env(
            env,
            "RECEIPT_DB_PATH",
            database_dir / "receipt_intelligence.db",
        ).resolve()

        return cls(
            project_root=root,
            var_root=var_root,
            uploads_dir=uploads_dir,
            jobs_dir=jobs_dir,
            database_dir=database_dir,
            reports_dir=reports_dir,
            batch_input_dir=batch_input_dir,
            logs_dir=logs_dir,
            receipt_db_path=receipt_db_path,
        )

    @property
    def layout(self) -> str:
        """Return the stable layout name retained in the public config API."""

        return "var"

    @property
    def job_read_roots(self) -> tuple[Path, ...]:
        return (self.jobs_dir,)

    @property
    def batch_read_roots(self) -> tuple[Path, ...]:
        return (self.batch_input_dir,)

    def ensure_directories(self) -> None:
        for path in (
            self.var_root,
            self.uploads_dir,
            self.jobs_dir,
            self.database_dir,
            self.reports_dir,
            self.batch_input_dir,
            self.logs_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)

    def as_dict(self) -> dict[str, object]:
        return {
            "layout": self.layout,
            "project_root": str(self.project_root),
            "var_root": str(self.var_root),
            "uploads_dir": str(self.uploads_dir),
            "jobs_dir": str(self.jobs_dir),
            "database_dir": str(self.database_dir),
            "reports_dir": str(self.reports_dir),
            "batch_input_dir": str(self.batch_input_dir),
            "logs_dir": str(self.logs_dir),
            "receipt_db_path": str(self.receipt_db_path),
            "job_read_roots": [str(self.jobs_dir)],
            "batch_read_roots": [str(self.batch_input_dir)],
        }
