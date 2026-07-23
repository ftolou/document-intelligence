from __future__ import annotations

from receipt_intelligence.rag.candidate_models import CandidateRecord
from receipt_intelligence.rag.candidate_prompt import build_candidate_resolution_prompt


def test_prompt_uses_evidence_contract_and_marks_context_low_trust() -> None:
    prompt = build_candidate_resolution_prompt(
        "Schuhe",
        [
            CandidateRecord(
                candidate_id="c001",
                item_ids=[84],
                description="SCHUHENGEL, GRAU paar",
                normalized_description="schuhengel grau paar",
                occurrence_count=1,
                category="Food & Groceries / beverages",
                semantic_description="Vittel is a brand of mineral water.",
                merchant="LIDL",
            )
        ],
        user_question="Wie viel habe ich für Schuhe ausgegeben?",
    )

    assert '"candidate_id":"c001"' in prompt
    assert '"schema_version": "rag_candidate_resolution_v2"' in prompt
    assert "category_low_trust" in prompt
    assert "merchant_low_trust" in prompt
    assert "semantic_description_reviewed" in prompt
    assert "Vittel is a brand of mineral water" in prompt
    assert "Never select a candidate solely because its category or merchant" in prompt
    assert "Ambiguous product descriptions must be classified as uncertain" in prompt
    assert '"evidence_strength"' in prompt
    assert "Do not return confidence numbers" in prompt
    assert '"similarity":' not in prompt
    assert '"fusion_score":' not in prompt
    assert "Do not generate SQL" in prompt
    assert "Wie viel habe ich für Schuhe ausgegeben?" in prompt
