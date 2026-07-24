"""Bounded in-process background dispatcher with persistent job lifecycle state."""

from __future__ import annotations

import os
import socket
import threading
import traceback
import uuid
from concurrent.futures import Future, ThreadPoolExecutor

from receipt_intelligence.application.ports.jobs import (
    JobDispatcher,
    JobDispatchRequest,
    JobProcessor,
    JobQueueFullError,
    JobRepository,
)


class ThreadPoolJobDispatcher(JobDispatcher):
    """Run persisted job requests on a bounded process-level executor.

    Filesystem claims and a managed heartbeat prevent duplicate execution when
    multiple application processes share the same runtime directory. The
    coordinator also retries persisted queued/running jobs after an interrupted
    process once its claim lease expires.
    """

    def __init__(
        self,
        repository: JobRepository,
        processor: JobProcessor,
        *,
        max_workers: int = 1,
        queue_capacity: int = 32,
        claim_lease_seconds: float = 60.0,
        maintenance_interval_seconds: float = 10.0,
        thread_name_prefix: str = "receipt-job",
    ) -> None:
        if max_workers < 1:
            raise ValueError("max_workers must be at least 1")
        if queue_capacity < 0:
            raise ValueError("queue_capacity must not be negative")
        if maintenance_interval_seconds <= 0:
            raise ValueError("maintenance_interval_seconds must be positive")
        self._repository = repository
        self._processor = processor
        self._claim_lease_seconds = max(
            float(claim_lease_seconds),
            float(maintenance_interval_seconds) * 3.0,
        )
        self._maintenance_interval_seconds = float(maintenance_interval_seconds)
        self._owner_id = f"{socket.gethostname()}:{os.getpid()}:{uuid.uuid4().hex[:12]}"
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix=thread_name_prefix,
        )
        self._slots = threading.BoundedSemaphore(max_workers + queue_capacity)
        self._lock = threading.RLock()
        self._futures: dict[str, Future[None]] = {}
        self._active_claims: set[str] = set()
        self._shutdown = False
        self._stop_maintenance = threading.Event()
        self._maintenance_thread: threading.Thread | None = None

    @property
    def owner_id(self) -> str:
        return self._owner_id

    def submit(self, request: JobDispatchRequest) -> None:
        with self._lock:
            if self._shutdown:
                raise JobQueueFullError("Background job dispatcher is shutting down.")
            existing = self._futures.get(request.job_id)
            if existing is not None and not existing.done():
                return
            if not self._slots.acquire(blocking=False):
                raise JobQueueFullError("Background job queue is full.")
            try:
                future = self._executor.submit(self._execute, request)
            except Exception:
                self._slots.release()
                raise
            self._futures[request.job_id] = future
            future.add_done_callback(
                lambda completed, job_id=request.job_id: self._on_done(job_id, completed)
            )
            self._ensure_maintenance_thread()

    def recover_pending(self) -> int:
        recovered = 0
        dispatchable = self._repository.list_dispatchable()
        for job in dispatchable:
            job_id = str(job.get("job_id") or "")
            dispatch = job.get("dispatch")
            if not job_id or not isinstance(dispatch, dict):
                continue
            with self._lock:
                if job_id in self._active_claims:
                    continue
                current = self._futures.get(job_id)
                if current is not None and not current.done():
                    continue
            try:
                request = JobDispatchRequest.from_payload(job_id, dispatch)
                self.submit(request)
            except (ValueError, JobQueueFullError):
                continue
            recovered += 1
        if dispatchable:
            self._ensure_maintenance_thread()
        return recovered

    def shutdown(self, *, wait: bool = True, cancel_futures: bool = False) -> None:
        with self._lock:
            if self._shutdown:
                return
            self._shutdown = True
        self._executor.shutdown(wait=wait, cancel_futures=cancel_futures)
        self._stop_maintenance.set()
        thread = self._maintenance_thread
        if thread is not None and thread.is_alive() and thread is not threading.current_thread():
            thread.join(timeout=max(1.0, self._maintenance_interval_seconds * 2.0))

    def _execute(self, request: JobDispatchRequest) -> None:
        claimed = self._repository.try_claim(
            request.job_id,
            self._owner_id,
            lease_seconds=self._claim_lease_seconds,
        )
        if not claimed:
            return

        with self._lock:
            self._active_claims.add(request.job_id)
        try:
            self._repository.begin_attempt(request.job_id)
            self._repository.add_event(
                request.job_id,
                {
                    "stage": "worker",
                    "status": "running",
                    "message": "Background worker accepted the job.",
                    "details": {"worker_owner": self._owner_id},
                },
            )
            if request.kind == "receipt":
                assert request.image_path is not None
                self._processor.run_job(request.job_id, request.image_path, request.options)
            else:
                self._processor.run_batch_job(
                    request.job_id,
                    list(request.image_paths),
                    request.options,
                )
            self._repository.complete(request.job_id)
            self._repository.add_event(
                request.job_id,
                {
                    "stage": "worker",
                    "status": "done",
                    "message": "Background job completed.",
                },
            )
        except Exception as exc:
            traceback_text = traceback.format_exc()
            error = {
                "message": str(exc),
                "type": type(exc).__name__,
                "traceback": traceback_text,
            }
            self._repository.fail(request.job_id, error)
            self._repository.add_event(
                request.job_id,
                {
                    "stage": "worker",
                    "status": "error",
                    "message": str(exc),
                    "details": {"traceback": traceback_text[-4000:]},
                },
            )
        finally:
            with self._lock:
                self._active_claims.discard(request.job_id)
            self._repository.release_claim(request.job_id, self._owner_id)

    def _on_done(self, job_id: str, future: Future[None]) -> None:
        try:
            if future.cancelled():
                return
            unexpected = future.exception()
            if unexpected is not None:
                traceback_text = "".join(
                    traceback.format_exception(
                        type(unexpected),
                        unexpected,
                        unexpected.__traceback__,
                    )
                )
                self._repository.fail(
                    job_id,
                    {
                        "message": str(unexpected),
                        "type": type(unexpected).__name__,
                        "traceback": traceback_text,
                    },
                )
        finally:
            with self._lock:
                self._futures.pop(job_id, None)
            self._slots.release()

    def _ensure_maintenance_thread(self) -> None:
        with self._lock:
            if self._shutdown:
                return
            if self._maintenance_thread is not None and self._maintenance_thread.is_alive():
                return
            self._maintenance_thread = threading.Thread(
                target=self._maintenance_loop,
                name="receipt-job-coordinator",
                daemon=True,
            )
            self._maintenance_thread.start()

    def _maintenance_loop(self) -> None:
        while not self._stop_maintenance.wait(self._maintenance_interval_seconds):
            with self._lock:
                active = tuple(self._active_claims)
                shutting_down = self._shutdown
            for job_id in active:
                self._repository.renew_claim(
                    job_id,
                    self._owner_id,
                    lease_seconds=self._claim_lease_seconds,
                )
            if not shutting_down:
                self.recover_pending()


__all__ = ["ThreadPoolJobDispatcher"]
