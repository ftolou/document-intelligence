"""Small monotonic timing and UTC timestamp helpers."""

from __future__ import annotations

import time
from datetime import datetime, timezone


def utc_now_iso() -> str:
    """Return an RFC 3339 UTC timestamp with millisecond precision."""

    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def elapsed_ms(started_at: float) -> float:
    """Return elapsed monotonic time in milliseconds."""

    return round((time.perf_counter() - started_at) * 1000.0, 3)
