from __future__ import annotations

import pytest
from pydantic import ValidationError

from receipt_intelligence.rag.candidate_models import (
    CandidateDecision,
    CandidateResolutionPayload,
)


def test_resolved_payload_requires_selected_candidate() -> None:
    with pytest.raises(ValidationError, match="requires at least one selected"):
        CandidateResolutionPayload.model_validate(
            {
                "schema_version": "rag_candidate_resolution_v2",
                "status": "resolved",
                "semantic_entity": "Schuhe",
                "decisions": [
                    {
                        "candidate_id": "c001",
                        "decision": "rejected",
                        "evidence_strength": "unrelated",
                        "evidence": "The description denotes a different product.",
                    }
                ],
                "clarification_question": None,
                "notes": [],
            }
        )


def test_needs_clarification_requires_uncertain_candidate_and_question() -> None:
    payload = CandidateResolutionPayload.model_validate(
        {
            "schema_version": "rag_candidate_resolution_v2",
            "status": "needs_clarification",
            "semantic_entity": "Schuhe",
            "decisions": [
                {
                    "candidate_id": "c001",
                    "decision": "uncertain",
                    "evidence_strength": "ambiguous",
                    "evidence": "The description could denote a shoe accessory.",
                }
            ],
            "clarification_question": "Should shoe accessories be included?",
            "notes": [],
        }
    )

    assert payload.status == "needs_clarification"


@pytest.mark.parametrize(
    ("decision", "evidence_strength"),
    [
        ("selected", "ambiguous"),
        ("selected", "unrelated"),
        ("uncertain", "explicit"),
        ("uncertain", "strong_contextual"),
        ("rejected", "ambiguous"),
        ("rejected", "explicit"),
    ],
)
def test_decision_requires_consistent_evidence_strength(
    decision: str,
    evidence_strength: str,
) -> None:
    with pytest.raises(ValidationError, match="inconsistent"):
        CandidateDecision.model_validate(
            {
                "candidate_id": "c001",
                "decision": decision,
                "evidence_strength": evidence_strength,
                "evidence": "Test evidence.",
            }
        )


@pytest.mark.parametrize(
    ("decision", "evidence_strength"),
    [
        ("selected", "explicit"),
        ("selected", "strong_contextual"),
        ("uncertain", "ambiguous"),
        ("rejected", "unrelated"),
    ],
)
def test_valid_decision_evidence_mappings(
    decision: str,
    evidence_strength: str,
) -> None:
    result = CandidateDecision.model_validate(
        {
            "candidate_id": "c001",
            "decision": decision,
            "evidence_strength": evidence_strength,
            "evidence": "Description-based evidence.",
        }
    )

    assert result.evidence_strength == evidence_strength


def test_legacy_confidence_contract_is_rejected() -> None:
    with pytest.raises(ValidationError):
        CandidateDecision.model_validate(
            {
                "candidate_id": "c001",
                "decision": "selected",
                "confidence": 1.0,
                "reason": "Legacy output.",
            }
        )
