from __future__ import annotations

from pathlib import Path

import pytest

from receipt_intelligence.runtime.paths import RuntimePaths


def test_runtime_paths_use_only_canonical_var_root(tmp_path: Path) -> None:
    paths = RuntimePaths.from_environment(tmp_path, {"VAR_DIR": str(tmp_path / "state")})

    assert paths.layout == "var"
    assert paths.jobs_dir == (tmp_path / "state" / "jobs").resolve()
    assert (
        paths.receipt_db_path
        == (tmp_path / "state" / "database" / "receipt_intelligence.db").resolve()
    )
    assert paths.job_read_roots == (paths.jobs_dir,)
    assert paths.batch_read_roots == (paths.batch_input_dir,)
    assert "legacy_read_enabled" not in paths.as_dict()


def test_runtime_paths_reject_legacy_layout(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Only the canonical var runtime layout"):
        RuntimePaths.from_environment(tmp_path, {"RUNTIME_LAYOUT": "legacy"})


def test_ensure_directories_does_not_copy_legacy_database(tmp_path: Path) -> None:
    legacy_database = tmp_path / "data" / "receipt_intelligence.db"
    legacy_database.parent.mkdir(parents=True)
    legacy_database.write_bytes(b"legacy-db")

    paths = RuntimePaths.from_environment(tmp_path, environ={})
    paths.ensure_directories()

    assert not paths.receipt_db_path.exists()
    assert paths.database_dir.exists()
