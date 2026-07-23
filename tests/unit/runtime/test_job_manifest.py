from __future__ import annotations

from pathlib import Path

import pytest

from receipt_intelligence.runtime.manifest import JobManifestStore


def test_manifest_registers_relative_artifact_metadata(tmp_path: Path) -> None:
    job_dir = tmp_path / "jobs" / "abc"
    artifact = job_dir / "receipt_final.json"
    artifact.parent.mkdir(parents=True)
    artifact.write_text("{}", encoding="utf-8")

    store = JobManifestStore()
    entry = store.register_artifact(job_dir, "abc", "final_receipt", artifact)
    manifest = store.load(job_dir)

    assert entry["path"] == "receipt_final.json"
    assert entry["category"] == "final"
    assert manifest is not None
    assert manifest["schema_version"] == "job_manifest_v1"
    assert manifest["artifacts"]["final_receipt"]["size_bytes"] == 2


def test_manifest_rejects_artifacts_outside_job_directory(tmp_path: Path) -> None:
    job_dir = tmp_path / "jobs" / "abc"
    outside = tmp_path / "outside.json"
    outside.write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="outside job directory"):
        JobManifestStore().register_artifact(job_dir, "abc", "outside", outside)
