"""Thread-safe JSON Lines persistence for compact runtime events."""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any


class JsonlEventWriter:
    """Append one JSON object per line.

    The writer is deliberately simple: local append-only telemetry must never be
    able to stop receipt extraction or querying. Callers decide whether to ignore
    write errors or surface them in diagnostics.
    """

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self._lock = threading.Lock()

    def append(self, event: dict[str, Any]) -> None:
        payload = json.dumps(event, ensure_ascii=False, separators=(",", ":"))
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(payload)
                handle.write("\n")
