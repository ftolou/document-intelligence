"""Runtime information use case."""

from __future__ import annotations

from typing import Any

from receipt_intelligence.application.ports.runtime import RuntimeInformation


class RuntimeUseCases:
    def __init__(self, runtime_information: RuntimeInformation) -> None:
        self._runtime_information = runtime_information

    def health(self) -> dict[str, Any]:
        return self._runtime_information.health()

    def readiness(self) -> dict[str, Any]:
        return self._runtime_information.readiness()

    def configuration(self) -> dict[str, Any]:
        return self._runtime_information.configuration()


__all__ = ["RuntimeUseCases"]
