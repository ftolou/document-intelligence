from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from receipt_intelligence.prompts.registry import (
    PromptIntegrityError,
    PromptReference,
    PromptRegistry,
    PromptRenderError,
)


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _write_registry(tmp_path: Path) -> PromptReference:
    template = "Receipt rows:\n{{ROWS}}\n"
    schema_text = json.dumps({"type": "object"}, indent=2) + "\n"
    (tmp_path / "template.txt").write_text(template, encoding="utf-8")
    (tmp_path / "schema.json").write_text(schema_text, encoding="utf-8")
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "prompt_manifest.v1",
                "prompts": [
                    {
                        "id": "gemma.test.receipt",
                        "version": "1.0.0",
                        "template": "template.txt",
                        "sha256": _digest(template),
                        "schema": "schema.json",
                        "schema_sha256": _digest(schema_text),
                        "required_variables": ["ROWS"],
                    }
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return PromptReference("gemma.test.receipt", "1.0.0")


def test_registry_verifies_and_renders_prompt_artifacts(tmp_path: Path) -> None:
    reference = _write_registry(tmp_path)
    registry = PromptRegistry(tmp_path)

    assert registry.render(reference, ROWS="R0001 :: SUMME 12,34").endswith(
        "R0001 :: SUMME 12,34\n"
    )
    assert registry.load_schema(reference) == {"type": "object"}


def test_registry_rejects_missing_variables(tmp_path: Path) -> None:
    reference = _write_registry(tmp_path)
    registry = PromptRegistry(tmp_path)

    with pytest.raises(PromptRenderError, match="ROWS"):
        registry.render(reference)


def test_registry_rejects_modified_prompt(tmp_path: Path) -> None:
    reference = _write_registry(tmp_path)
    registry = PromptRegistry(tmp_path)
    (tmp_path / "template.txt").write_text("modified", encoding="utf-8")

    with pytest.raises(PromptIntegrityError, match="hash mismatch"):
        registry.read_template(reference)
