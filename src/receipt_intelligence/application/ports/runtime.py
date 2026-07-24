"""Port for runtime health, readiness, and configuration information."""

from __future__ import annotations

from typing import Any, Protocol


class RuntimeInformation(Protocol):
    def health(self) -> dict[str, Any]: ...

    def readiness(self) -> dict[str, Any]: ...

    def configuration(self) -> dict[str, Any]: ...


__all__ = ["RuntimeInformation"]
