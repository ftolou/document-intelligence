"""Application composition helpers selecting infrastructure adapters."""

from __future__ import annotations

from receipt_intelligence.adapters.jobs import ThreadPoolJobDispatcher
from receipt_intelligence.application.ports import (
    JobDispatcher,
    JobProcessor,
    JobRepository,
)


def build_job_dispatcher(
    repository: JobRepository,
    processor: JobProcessor,
    *,
    max_workers: int = 1,
    queue_capacity: int = 32,
    claim_lease_seconds: float = 120.0,
    maintenance_interval_seconds: float = 10.0,
) -> JobDispatcher:
    return ThreadPoolJobDispatcher(
        repository,
        processor,
        max_workers=max_workers,
        queue_capacity=queue_capacity,
        claim_lease_seconds=claim_lease_seconds,
        maintenance_interval_seconds=maintenance_interval_seconds,
    )


__all__ = ["build_job_dispatcher"]
