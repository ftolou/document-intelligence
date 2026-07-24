"""Provider-neutral application event contracts."""

from __future__ import annotations

from typing import Any, Protocol


class ApplicationEvent(Protocol):
    """Immutable event that can be serialized by an outer adapter."""

    event_name: str
    occurred_at: str

    def to_record(self) -> dict[str, Any]: ...


class EventSink(Protocol):
    """Publish application events without exposing an observability backend."""

    def publish(self, event: ApplicationEvent) -> None: ...


class NullEventSink:
    """Default sink used when event persistence is disabled."""

    def publish(self, event: ApplicationEvent) -> None:
        del event


__all__ = ["ApplicationEvent", "EventSink", "NullEventSink"]
