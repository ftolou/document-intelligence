"""Architecture guards and regression cover for the validation trust boundary.

These tests hold on the current repository and must keep holding once the
deterministic validation boundary exists.
"""

from __future__ import annotations

import ast
from pathlib import Path

import interpretation_outcome_support as support
import pytest

from receipt_intelligence.application.ports.llm import (
    GenerationError,
    GenerationProviderUnavailableError,
    MalformedGenerationError,
)
from receipt_intelligence.interpretation import EvidenceReference, OnePassDocumentInterpreter

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src" / "receipt_intelligence"
INTERPRETATION = SRC / "interpretation"

_FORBIDDEN_IMPORT_PREFIXES = (
    "receipt_intelligence.storage",
    "receipt_intelligence.web",
    "receipt_intelligence.services",
    "receipt_intelligence.rag",
    "receipt_intelligence.rag_sql",
    "receipt_intelligence.adapters",
    "receipt_intelligence.application.use_cases",
    "sqlite3",
    "flask",
    "requests",
    "openai",
)
_FORBIDDEN_CURRENT_STATE_CALLS = ("datetime.now", "date.today", "utcnow", "time.time(")


def _imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            modules.add(node.module)
    return modules


def test_interpretation_package_stays_core_pure() -> None:
    violations: list[str] = []
    for path in INTERPRETATION.rglob("*.py"):
        relative = str(path.relative_to(ROOT))
        for module in _imported_modules(path):
            if module.startswith(_FORBIDDEN_IMPORT_PREFIXES):
                violations.append(f"{relative}: imports {module}")
        text = path.read_text(encoding="utf-8-sig")
        violations.extend(
            f"{relative}: uses {pattern}"
            for pattern in _FORBIDDEN_CURRENT_STATE_CALLS
            if pattern in text
        )
    assert violations == []


def test_receipt_extraction_does_not_depend_on_document_interpretation() -> None:
    violations = [
        str(path.relative_to(ROOT))
        for path in (SRC / "extraction").rglob("*.py")
        for module in _imported_modules(path)
        if module.startswith("receipt_intelligence.interpretation")
    ]
    assert violations == []


def test_evidence_reference_exposes_no_independently_verified_text() -> None:
    field_names = set(EvidenceReference.model_fields)
    assert "excerpt" in field_names
    assert [name for name in field_names if "verif" in name] == []


def test_contract_level_invariants_are_not_re_expressed_as_validation_findings(
    tmp_path: Path,
) -> None:
    response = support.generated_response()
    response["candidate_facts"][0]["evidence_refs"] = ["e-unknown"]
    source_path, media_type = support.write_source(tmp_path)
    interpreter = OnePassDocumentInterpreter(
        gateway=support.RecordingGateway(response),
        model="generic-multimodal-model",
        source_limits=support.limits(),
    )

    with pytest.raises(MalformedGenerationError):
        interpreter.interpret(support.interpretation_request(media_type=media_type), source_path)


@pytest.mark.parametrize(
    ("response", "expected"),
    [
        pytest.param("not json", MalformedGenerationError, id="malformed-json"),
        pytest.param({"classification": {}}, MalformedGenerationError, id="schema-invalid"),
    ],
)
def test_generation_failure_semantics_survive_the_outcome_return_type(
    tmp_path: Path,
    response: object,
    expected: type[Exception],
) -> None:
    source_path, media_type = support.write_source(tmp_path)
    interpreter = OnePassDocumentInterpreter(
        gateway=support.RecordingGateway(response),  # type: ignore[arg-type]
        model="generic-multimodal-model",
        source_limits=support.limits(),
    )

    with pytest.raises(expected):
        interpreter.interpret(support.interpretation_request(media_type=media_type), source_path)


def test_provider_failures_are_not_absorbed_by_deterministic_validation(tmp_path: Path) -> None:
    error = GenerationProviderUnavailableError("provider unavailable")

    class _FailingGateway:
        def generate(self, request: object) -> object:
            raise error

    source_path, media_type = support.write_source(tmp_path)
    interpreter = OnePassDocumentInterpreter(
        gateway=_FailingGateway(),  # type: ignore[arg-type]
        model="generic-multimodal-model",
        source_limits=support.limits(),
    )

    with pytest.raises(GenerationError) as raised:
        interpreter.interpret(support.interpretation_request(media_type=media_type), source_path)

    assert raised.value is error
