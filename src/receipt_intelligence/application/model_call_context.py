"""Request-scoped metadata attached to model-call telemetry."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass, replace
from typing import Iterator


@dataclass(frozen=True, slots=True)
class ModelCallContext:
    """Identifiers shared by model calls within one application operation."""

    trace_id: str | None = None
    job_id: str | None = None
    receipt_id: str | None = None
    query_id: str | None = None

    def merged(self, other: ModelCallContext | None) -> ModelCallContext:
        if other is None:
            return self
        return replace(
            self,
            trace_id=other.trace_id or self.trace_id,
            job_id=other.job_id or self.job_id,
            receipt_id=other.receipt_id or self.receipt_id,
            query_id=other.query_id or self.query_id,
        )


_CURRENT_MODEL_CALL_CONTEXT: ContextVar[ModelCallContext] = ContextVar(
    "receipt_intelligence_model_call_context",
    default=ModelCallContext(),
)


def current_model_call_context() -> ModelCallContext:
    return _CURRENT_MODEL_CALL_CONTEXT.get()


@contextmanager
def bind_model_call_context(
    *,
    trace_id: str | None = None,
    job_id: str | None = None,
    receipt_id: str | None = None,
    query_id: str | None = None,
) -> Iterator[ModelCallContext]:
    """Temporarily enrich model-call events for the current execution context."""

    current = current_model_call_context()
    updated = current.merged(
        ModelCallContext(
            trace_id=trace_id,
            job_id=job_id,
            receipt_id=receipt_id,
            query_id=query_id,
        )
    )
    token: Token[ModelCallContext] = _CURRENT_MODEL_CALL_CONTEXT.set(updated)
    try:
        yield updated
    finally:
        _CURRENT_MODEL_CALL_CONTEXT.reset(token)


__all__ = [
    "ModelCallContext",
    "bind_model_call_context",
    "current_model_call_context",
]
