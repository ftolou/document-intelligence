from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def test_removed_qa_alias_package_is_not_present() -> None:
    assert not (ROOT / "src" / "receipt_intelligence" / "qa").exists()
    assert importlib.util.find_spec("receipt_intelligence.qa") is None


def test_compose_uses_only_var_runtime_mount() -> None:
    for filename in ("docker-compose.yml", "docker-compose.dev.yml"):
        text = (ROOT / filename).read_text(encoding="utf-8")
        assert "./var:/app/var" in text
        assert "./uploads:/app/uploads" not in text
        assert "./outputs:/app/outputs" not in text
        assert "./data:/app/data" not in text
        assert "./batch_input:/app/batch_input" not in text
        assert "RUNTIME_LEGACY" not in text


def test_superseded_docker_wrappers_are_absent() -> None:
    obsolete = (
        "build_all_images.ps1",
        "build_runtime_images.ps1",
        "build_thin_images.ps1",
        "restart_all.ps1",
        "restart_app.ps1",
        "restart_vlm.ps1",
        "start_dev.ps1",
    )
    for filename in obsolete:
        assert not (ROOT / "scripts" / filename).exists()
