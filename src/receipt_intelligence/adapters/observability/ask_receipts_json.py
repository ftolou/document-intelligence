"""Atomic JSON artifact writer for opt-in Ask Your Receipts diagnostics."""

from __future__ import annotations

import json
import os
import re
import tempfile
import threading
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


class AskReceiptsJsonLogWriter:
    """Write one standalone JSON document per query without affecting execution."""

    def __init__(self, directory: Path | str, *, enabled: bool = True) -> None:
        self.directory = Path(directory)
        self.enabled = bool(enabled)
        self._lock = threading.Lock()

    def write(self, record: Mapping[str, Any], *, log_id: str) -> str | None:
        if not self.enabled:
            return None

        safe_log_id = re.sub(r"[^A-Za-z0-9._-]+", "_", str(log_id or "query"))[:120]
        timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ")
        filename = f"{timestamp}_{safe_log_id}.json"
        path = self.directory / filename

        try:
            with self._lock:
                self._write_atomic(path, dict(record))
        except (OSError, TypeError, ValueError):
            return None
        return filename

    @staticmethod
    def _write_atomic(path: Path, record: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
        )
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
                json.dump(record, handle, ensure_ascii=False, indent=2, default=str)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_name, path)
        except Exception:
            try:
                os.unlink(temporary_name)
            except OSError:
                pass
            raise


__all__ = ["AskReceiptsJsonLogWriter"]
