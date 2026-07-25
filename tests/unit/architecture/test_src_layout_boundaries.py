"""Regression checks for the repository's src-layout boundary."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def test_package_like_root_directories_do_not_duplicate_src_packages() -> None:
    package_root = ROOT / "src" / "receipt_intelligence"
    duplicate_directories = sorted(
        path.name
        for path in ROOT.iterdir()
        if path.is_dir() and (package_root / path.name).is_dir() and any(path.glob("*.py"))
    )

    assert duplicate_directories == []
