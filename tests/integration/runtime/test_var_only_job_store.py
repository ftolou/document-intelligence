from __future__ import annotations

import json
from pathlib import Path

from receipt_intelligence.storage.job_store import JobStore


def test_job_store_reads_and_writes_only_primary_root(tmp_path: Path) -> None:
    primary = tmp_path / "var" / "jobs"
    legacy = tmp_path / "outputs" / "results" / "oldjob"
    legacy.mkdir(parents=True)
    (legacy / "job_status.json").write_text(
        json.dumps({"job_id": "oldjob", "state": "done"}),
        encoding="utf-8",
    )

    store = JobStore(primary)
    assert store.get("oldjob") is None

    store.create("newjob", {"filename": "new.jpg"})
    assert (primary / "newjob" / "job_status.json").exists()
    assert (primary / "newjob" / "manifest.json").exists()
    assert store.read_roots == (primary.resolve(),)


def test_job_store_persists_execution_lifecycle(tmp_path: Path) -> None:
    root = tmp_path / "var" / "jobs"
    store = JobStore(root)
    store.create("lifecycle", {"dispatch": {"kind": "receipt"}})

    queued = store.get("lifecycle") or {}
    assert queued["state"] == "queued"
    assert queued["attempt_count"] == 0
    assert queued["started_at"] is None
    assert queued["finished_at"] is None

    store.begin_attempt("lifecycle")
    running = JobStore(root).get("lifecycle") or {}
    assert running["state"] == "running"
    assert running["attempt_count"] == 1
    assert running["started_at"]

    store.complete("lifecycle")
    completed = JobStore(root).get("lifecycle") or {}
    assert completed["state"] == "completed"
    assert completed["finished_at"]
    assert completed["error"] is None
