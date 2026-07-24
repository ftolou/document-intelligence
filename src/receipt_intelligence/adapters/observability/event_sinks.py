"""Local event-sink adapters for JSON and JSON Lines persistence."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import threading
from collections.abc import Iterable
from pathlib import Path

from receipt_intelligence.application.ports.events import ApplicationEvent, EventSink


class JsonlEventSink:
    """Append application events to a JSONL file without affecting the use case."""

    def __init__(self, path: Path | str, *, enabled: bool = True) -> None:
        self.path = Path(path)
        self.enabled = bool(enabled)
        self._lock = threading.Lock()

    def publish(self, event: ApplicationEvent) -> None:
        if not self.enabled:
            return
        payload = json.dumps(event.to_record(), ensure_ascii=False, separators=(",", ":"))
        try:
            with self._lock:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with self.path.open("a", encoding="utf-8", newline="\n") as handle:
                    handle.write(payload)
                    handle.write("\n")
        except OSError:
            return


class JsonFileEventSink:
    """Persist the latest event snapshot atomically and update optional aliases."""

    def __init__(
        self,
        path: Path | str,
        *,
        aliases: Iterable[Path | str] = (),
        enabled: bool = True,
    ) -> None:
        self.path = Path(path)
        self.aliases = tuple(Path(alias) for alias in aliases)
        self.enabled = bool(enabled)
        self._lock = threading.Lock()

    def publish(self, event: ApplicationEvent) -> None:
        if not self.enabled:
            return
        record = event.to_record()
        try:
            with self._lock:
                _write_json_atomic(self.path, record)
                for alias in self.aliases:
                    alias.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(self.path, alias)
        except OSError:
            return


class CompositeEventSink:
    """Fan out an event to multiple independent sinks."""

    def __init__(self, sinks: Iterable[EventSink]) -> None:
        self.sinks = tuple(sinks)

    def publish(self, event: ApplicationEvent) -> None:
        for sink in self.sinks:
            sink.publish(event)


def _write_json_atomic(path: Path, record: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    try:
        with os.fdopen(file_descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(record, handle, ensure_ascii=False, indent=2)
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


__all__ = ["CompositeEventSink", "JsonFileEventSink", "JsonlEventSink"]
