"""Behavioral tests for bounded persistent background execution."""

from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest

from receipt_intelligence.adapters.jobs import ThreadPoolJobDispatcher
from receipt_intelligence.application.ports.jobs import JobDispatchRequest, JobQueueFullError
from receipt_intelligence.storage.job_store import JobStore


def _wait_for_state(store: JobStore, job_id: str, expected: str, timeout: float = 3.0) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        job = store.get(job_id) or {}
        if job.get("state") == expected:
            return job
        time.sleep(0.01)
    raise AssertionError(f"Job {job_id} did not reach {expected}: {store.get(job_id)}")


def _receipt_request(store: JobStore, job_id: str) -> JobDispatchRequest:
    image_path = store.job_dir(job_id) / "receipt.jpg"
    image_path.parent.mkdir(parents=True, exist_ok=True)
    image_path.write_bytes(b"image")
    request = JobDispatchRequest(
        job_id=job_id,
        kind="receipt",
        image_path=image_path,
        options={"test": True},
    )
    store.create(
        job_id,
        {
            "filename": image_path.name,
            "image_path": str(image_path),
            "dispatch": request.to_payload(),
        },
    )
    return request


class _Processor:
    def __init__(
        self,
        *,
        error: Exception | None = None,
        gate: threading.Event | None = None,
    ) -> None:
        self.error = error
        self.gate = gate
        self.calls: list[str] = []
        self.started = threading.Event()

    def run_job(self, job_id: str, _image_path: Path, _options: dict) -> None:
        self.calls.append(job_id)
        self.started.set()
        if self.gate is not None:
            self.gate.wait(timeout=2.0)
        if self.error is not None:
            raise self.error

    def run_batch_job(self, *_args) -> None:
        raise AssertionError("Unexpected batch execution")


def test_dispatcher_persists_successful_lifecycle(tmp_path: Path) -> None:
    store = JobStore(tmp_path / "jobs")
    request = _receipt_request(store, "success")
    processor = _Processor()
    dispatcher = ThreadPoolJobDispatcher(
        store,
        processor,  # type: ignore[arg-type]
        maintenance_interval_seconds=0.05,
        claim_lease_seconds=1.0,
    )
    try:
        dispatcher.submit(request)
        job = _wait_for_state(store, request.job_id, "completed")
    finally:
        dispatcher.shutdown()

    assert processor.calls == ["success"]
    assert job["attempt_count"] == 1
    assert job["started_at"]
    assert job["finished_at"]
    assert job["error"] is None


def test_dispatcher_persists_worker_failure(tmp_path: Path) -> None:
    store = JobStore(tmp_path / "jobs")
    request = _receipt_request(store, "failure")
    dispatcher = ThreadPoolJobDispatcher(
        store,
        _Processor(error=RuntimeError("pipeline failed")),  # type: ignore[arg-type]
        maintenance_interval_seconds=0.05,
        claim_lease_seconds=1.0,
    )
    try:
        dispatcher.submit(request)
        job = _wait_for_state(store, request.job_id, "failed")
    finally:
        dispatcher.shutdown()

    assert job["attempt_count"] == 1
    assert job["error"]["type"] == "RuntimeError"
    assert job["error"]["message"] == "pipeline failed"


def test_dispatcher_recovers_persisted_queued_job(tmp_path: Path) -> None:
    store = JobStore(tmp_path / "jobs")
    request = _receipt_request(store, "recover")
    processor = _Processor()
    dispatcher = ThreadPoolJobDispatcher(
        store,
        processor,  # type: ignore[arg-type]
        maintenance_interval_seconds=0.05,
        claim_lease_seconds=1.0,
    )
    try:
        assert dispatcher.recover_pending() == 1
        _wait_for_state(store, request.job_id, "completed")
    finally:
        dispatcher.shutdown()

    assert processor.calls == ["recover"]


def test_dispatcher_rejects_work_beyond_bounded_capacity(tmp_path: Path) -> None:
    store = JobStore(tmp_path / "jobs")
    first = _receipt_request(store, "first")
    second = _receipt_request(store, "second")
    gate = threading.Event()
    processor = _Processor(gate=gate)
    dispatcher = ThreadPoolJobDispatcher(
        store,
        processor,  # type: ignore[arg-type]
        max_workers=1,
        queue_capacity=0,
        maintenance_interval_seconds=10.0,
        claim_lease_seconds=60.0,
    )
    try:
        dispatcher.submit(first)
        assert processor.started.wait(timeout=1.0)
        with pytest.raises(JobQueueFullError, match="queue is full"):
            dispatcher.submit(second)
    finally:
        gate.set()
        dispatcher.shutdown()


def test_filesystem_claim_prevents_duplicate_execution(tmp_path: Path) -> None:
    root = tmp_path / "jobs"
    first_store = JobStore(root)
    second_store = JobStore(root)
    request = _receipt_request(first_store, "claimed")
    gate = threading.Event()
    processor = _Processor(gate=gate)
    first = ThreadPoolJobDispatcher(
        first_store,
        processor,  # type: ignore[arg-type]
        maintenance_interval_seconds=10.0,
        claim_lease_seconds=60.0,
    )
    second = ThreadPoolJobDispatcher(
        second_store,
        processor,  # type: ignore[arg-type]
        maintenance_interval_seconds=10.0,
        claim_lease_seconds=60.0,
    )
    try:
        first.submit(request)
        assert processor.started.wait(timeout=1.0)
        second.submit(request)
        time.sleep(0.1)
        assert processor.calls == ["claimed"]
    finally:
        gate.set()
        first.shutdown()
        second.shutdown()

    assert (first_store.get("claimed") or {})["state"] == "completed"
