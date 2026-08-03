from __future__ import annotations

import hashlib
import json
from pathlib import Path

from receipt_intelligence.prompts.registry import PromptReference, PromptRegistry


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _write_registry(
    tmp_path: Path,
    *,
    template: str,
    required_variables: tuple[str, ...],
) -> PromptReference:
    (tmp_path / "template.txt").write_text(template, encoding="utf-8")
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "prompt_manifest.v1",
                "prompts": [
                    {
                        "id": "gemma.test.placeholders",
                        "version": "1.0.0",
                        "template": "template.txt",
                        "sha256": _digest(template),
                        "required_variables": list(required_variables),
                    }
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return PromptReference("gemma.test.placeholders", "1.0.0")


def test_registry_renders_all_supported_placeholder_styles(tmp_path: Path) -> None:
    reference = _write_registry(
        tmp_path,
        template=(
            "Mustache={{MUSTACHE}}\n"
            "Braced dollar=${BRACED_DOLLAR}\n"
            "Plain dollar=$PLAIN_DOLLAR\n"
            "Longer identifier stays=$PLAIN_DOLLAR_suffix\n"
        ),
        required_variables=("MUSTACHE", "BRACED_DOLLAR", "PLAIN_DOLLAR"),
    )

    rendered = PromptRegistry(tmp_path).render(
        reference,
        MUSTACHE="one",
        BRACED_DOLLAR="two",
        PLAIN_DOLLAR="three",
    )

    assert "Mustache=one" in rendered
    assert "Braced dollar=two" in rendered
    assert "Plain dollar=three" in rendered
    assert "$PLAIN_DOLLAR_suffix" in rendered


def test_registry_renders_gemma_task_envelope_evidence(tmp_path: Path) -> None:
    reference = _write_registry(
        tmp_path,
        template=(
            "$question\n\n"
            "Required JSON schema:\n"
            "$schema_json\n\n"
            "----- BEGIN TASK EVIDENCE -----\n"
            "$evidence\n"
            "----- END TASK EVIDENCE -----\n"
        ),
        required_variables=("question", "schema_json", "evidence"),
    )

    rendered = PromptRegistry(tmp_path).render(
        reference,
        question="Extract the final total.",
        schema_json='{"type":"object"}',
        evidence="R0001 :: Bonsumme 110,24",
    )

    assert "Extract the final total." in rendered
    assert '{"type":"object"}' in rendered
    assert "R0001 :: Bonsumme 110,24" in rendered
    assert "$question" not in rendered
    assert "$schema_json" not in rendered
    assert "$evidence" not in rendered


def test_inserted_values_are_not_rendered_again(tmp_path: Path) -> None:
    reference = _write_registry(
        tmp_path,
        template="$question\n$evidence\n",
        required_variables=("question", "evidence"),
    )

    rendered = PromptRegistry(tmp_path).render(
        reference,
        question="Explain the literal token $evidence.",
        evidence="R0001 :: TOTAL 9,99",
    )

    assert rendered == "Explain the literal token $evidence.\nR0001 :: TOTAL 9,99\n"
