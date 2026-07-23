#!/usr/bin/env python3
"""Fail when Requests emits a dependency compatibility warning."""

from __future__ import annotations

import importlib.metadata
import warnings

with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    import requests  # noqa: F401

compatibility_warnings = [
    warning for warning in caught if warning.category.__name__ == "RequestsDependencyWarning"
]

packages = ["requests", "urllib3", "charset-normalizer", "chardet"]
versions: dict[str, str | None] = {}
for package in packages:
    try:
        versions[package] = importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        versions[package] = None

print(versions)
if compatibility_warnings:
    for warning in compatibility_warnings:
        print(f"Requests dependency warning: {warning.message}")
    raise SystemExit(1)
print("Requests dependency versions are compatible.")
