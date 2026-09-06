"""Deterministic, provider-neutral planning for bounded interpretation work."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class InterpretationExecutionLimits:
    """Bounds for page images and generation windows used by one interpretation."""

    max_pages_per_window: int = 8
    max_windows: int = 8

    def __post_init__(self) -> None:
        if self.max_pages_per_window < 1:
            raise ValueError("InterpretationExecutionLimits.max_pages_per_window must be positive.")
        if self.max_windows < 1:
            raise ValueError("InterpretationExecutionLimits.max_windows must be positive.")


@dataclass(frozen=True, slots=True)
class InterpretationWindow:
    """One inclusive, one-based range of original source pages."""

    start_page: int
    end_page: int

    def __post_init__(self) -> None:
        if self.start_page < 1 or self.end_page < self.start_page:
            raise ValueError("InterpretationWindow must be a non-empty ordered page range.")

    @property
    def page_count(self) -> int:
        return self.end_page - self.start_page + 1


def plan_interpretation_windows(
    page_count: int,
    *,
    limits: InterpretationExecutionLimits,
) -> tuple[InterpretationWindow, ...]:
    """Partition all source pages into bounded, contiguous, non-overlapping windows."""

    if page_count < 1:
        raise ValueError("page_count must be positive.")

    window_count = (page_count + limits.max_pages_per_window - 1) // limits.max_pages_per_window
    if window_count > limits.max_windows:
        raise ValueError("Document interpretation requires more windows than the configured limit.")

    return tuple(
        InterpretationWindow(
            start_page=start,
            end_page=min(start + limits.max_pages_per_window - 1, page_count),
        )
        for start in range(1, page_count + 1, limits.max_pages_per_window)
    )


__all__ = [
    "InterpretationExecutionLimits",
    "InterpretationWindow",
    "plan_interpretation_windows",
]
