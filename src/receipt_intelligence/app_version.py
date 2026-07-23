"""Single source of truth for the receipt app version shown in API/UI/output metadata."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(os.getenv("APP_PROJECT_ROOT", Path.cwd())).resolve()
if not (PROJECT_ROOT / "VERSION").exists() and (PACKAGE_DIR.parents[1] / "VERSION").exists():
    PROJECT_ROOT = PACKAGE_DIR.parents[1]
VERSION_FILE = Path(os.getenv("APP_VERSION_FILE", PROJECT_ROOT / "VERSION"))
DEFAULT_APP_VERSION = "1.0.0"


def get_app_version() -> str:
    try:
        value = VERSION_FILE.read_text(encoding="utf-8").strip()
        return value or DEFAULT_APP_VERSION
    except Exception:
        return DEFAULT_APP_VERSION


def app_version_payload() -> dict[str, Any]:
    version = get_app_version()
    return {
        "version": version,
        "app_version": version,
        "version_file": str(VERSION_FILE),
    }
