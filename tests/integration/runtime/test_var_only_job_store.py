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
