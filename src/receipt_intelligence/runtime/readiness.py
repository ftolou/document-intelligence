"""Readiness checks for local and container deployments."""

from __future__ import annotations

import tempfile
from typing import Any

import requests

from receipt_intelligence.observability.timing import utc_now_iso
from receipt_intelligence.runtime.paths import RuntimePaths
from receipt_intelligence.storage.migrations import LATEST_SCHEMA_VERSION
from receipt_intelligence.storage.receipt_db import ReceiptDatabase


def _database_check(database: ReceiptDatabase) -> dict[str, Any]:
    try:
        with database.connect() as connection:
            connection.execute("SELECT 1").fetchone()
        version = database.migrations.current_version()
        ready = version == LATEST_SCHEMA_VERSION
        return {
            "status": "ok" if ready else "error",
            "required": True,
            "schema_version": version,
            "expected_schema_version": LATEST_SCHEMA_VERSION,
        }
    except Exception as exc:
        return {
            "status": "error",
            "required": True,
            "error": f"{type(exc).__name__}: {exc}",
        }


def _runtime_storage_check(paths: RuntimePaths) -> dict[str, Any]:
    try:
        paths.ensure_directories()
        with tempfile.NamedTemporaryFile(
            prefix="readiness_",
            suffix=".tmp",
            dir=paths.logs_dir,
            delete=True,
        ):
            pass
        return {
            "status": "ok",
            "required": True,
            "var_root": str(paths.var_root),
        }
    except Exception as exc:
        return {
            "status": "error",
            "required": True,
            "error": f"{type(exc).__name__}: {exc}",
        }


def _http_check(
    *,
    name: str,
    url: str,
    required: bool,
    enabled: bool,
    timeout_seconds: float,
) -> dict[str, Any]:
    if not enabled:
        return {
            "status": "skipped",
            "required": required,
            "url": url,
        }
    try:
        response = requests.get(url, timeout=timeout_seconds)
        response.raise_for_status()
        return {
            "status": "ok",
            "required": required,
            "url": url,
            "http_status": response.status_code,
        }
    except Exception as exc:
        return {
            "status": "error" if required else "unavailable_optional",
            "required": required,
            "url": url,
            "error": f"{type(exc).__name__}: {exc}",
            "service": name,
        }


def build_readiness_report(
    *,
    database: ReceiptDatabase,
    runtime_paths: RuntimePaths,
    ollama_url: str,
    probe_ollama: bool,
    require_ollama: bool,
    timeout_seconds: float,
) -> dict[str, Any]:
    """Build a deterministic readiness report.

    Database and writable runtime storage are always required. Ollama may be
    optional for selected endpoints.
    """

    checks = {
        "database": _database_check(database),
        "runtime_storage": _runtime_storage_check(runtime_paths),
        "ollama": _http_check(
            name="ollama",
            url=f"{ollama_url.rstrip('/')}/api/tags",
            required=require_ollama,
            enabled=probe_ollama or require_ollama,
            timeout_seconds=timeout_seconds,
        ),
    }
    ready = all(
        check.get("status") == "ok" for check in checks.values() if check.get("required") is True
    )
    degraded = any(check.get("status") == "unavailable_optional" for check in checks.values())
    return {
        "ready": ready,
        "degraded": degraded,
        "checked_at": utc_now_iso(),
        "checks": checks,
    }
