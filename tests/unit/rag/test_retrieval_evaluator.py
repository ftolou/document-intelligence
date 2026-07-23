from __future__ import annotations

import json
from pathlib import Path

import pytest

from receipt_intelligence.rag.models import (
    RetrievalEvaluationCase,
    SemanticItemMatch,
    SemanticItemSearchResult,
)
from receipt_intelligence.rag.retrieval_evaluator import (
    ItemRetrievalEvaluator,
    load_evaluation_cases,
)


class FakeRetriever:
    def __init__(self, results: dict[str, SemanticItemSearchResult]) -> None:
        self.results = results

    def search(
        self,
        query: str,
        *,
        limit: int = 10,
        minimum_score: float | None = None,
    ) -> SemanticItemSearchResult:
        return self.results[query]


def _result(query: str, matches: list[SemanticItemMatch]) -> SemanticItemSearchResult:
    return SemanticItemSearchResult(
        query=query,
        model="test-model",
        dimension=2,
        total_candidates=len(matches),
        limit=5,
        matches=matches,
    )


def _match(
    item_id: int,
    description: str,
    *,
    rank: int,
    parser_item_type: str = "item",
) -> SemanticItemMatch:
    return SemanticItemMatch(
        rank=rank,
        item_id=item_id,
        receipt_id=1,
        description=description,
        parser_item_type=parser_item_type,
        similarity=1.0 - rank / 10,
    )


def test_evaluator_reports_recall_reciprocal_rank_and_forbidden_types() -> None:
    retriever = FakeRetriever(
        {
            "Schuhe": _result(
                "Schuhe",
                [
                    _match(7, "UNKNOWN", rank=1),
                    _match(3, "DAMEN SNEAKER", rank=2),
                ],
            ),
            "Wasser": _result(
                "Wasser",
                [_match(9, "AKTIONSRABATT", rank=1, parser_item_type="discount")],
            ),
        }
    )
    cases = [
        RetrievalEvaluationCase(
            case_id="shoes",
            query="Schuhe",
            expected_item_ids=[3],
            expected_any_terms=["sneaker"],
        ),
        RetrievalEvaluationCase(
            case_id="water",
            query="Wasser",
            expected_any_terms=["wasser", "mineral"],
        ),
    ]

    report = ItemRetrievalEvaluator(retriever).evaluate(cases)

    assert report.case_count == 2
    assert report.passed == 1
    assert report.failed == 1
    assert report.mean_reciprocal_rank == pytest.approx(0.5)
    assert report.mean_recall_at_k == pytest.approx(1.0)
    assert report.mean_precision_at_k == pytest.approx(0.5)
    assert report.results[0].returned_identity_count == 2
    assert "Forbidden parser item types" in " ".join(report.results[1].errors)


def test_load_evaluation_cases_validates_json(tmp_path: Path) -> None:
    path = tmp_path / "cases.json"
    path.write_text(
        json.dumps(
            [
                {
                    "case_id": "toothpaste",
                    "query": "Zahnpasta",
                    "expected_any_terms": ["elmex", "zahnp"],
                }
            ]
        ),
        encoding="utf-8",
    )

    cases = load_evaluation_cases(path)

    assert cases[0].case_id == "toothpaste"
    assert cases[0].top_k == 5


def test_case_requires_an_expected_signal() -> None:
    with pytest.raises(ValueError, match="requires expected_item_ids"):
        RetrievalEvaluationCase(case_id="invalid", query="anything")
