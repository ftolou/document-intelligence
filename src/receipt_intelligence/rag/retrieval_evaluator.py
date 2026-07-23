"""Evaluation utilities for hybrid receipt-item retrieval."""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Protocol

from pydantic import TypeAdapter

from receipt_intelligence.rag.models import (
    RetrievalEvaluationCase,
    RetrievalEvaluationCaseResult,
    RetrievalEvaluationReport,
    SemanticItemSearchResult,
)


class SemanticRetriever(Protocol):
    def search(
        self,
        query: str,
        *,
        limit: int = 10,
        minimum_score: float | None = None,
    ) -> SemanticItemSearchResult: ...


def load_evaluation_cases(path: Path | str) -> list[RetrievalEvaluationCase]:
    """Load and strictly validate a JSON list of retrieval cases."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return TypeAdapter(list[RetrievalEvaluationCase]).validate_python(payload)


class ItemRetrievalEvaluator:
    def __init__(self, retriever: SemanticRetriever) -> None:
        self.retriever = retriever

    def evaluate(
        self,
        cases: Iterable[RetrievalEvaluationCase],
    ) -> RetrievalEvaluationReport:
        results: list[RetrievalEvaluationCaseResult] = []

        for case in cases:
            search_result = self.retriever.search(
                case.query,
                limit=case.top_k,
                minimum_score=case.minimum_score,
            )
            results.append(self._evaluate_case(case, search_result))

        passed = sum(1 for result in results if result.passed)
        reciprocal_ranks = [
            result.reciprocal_rank for result in results if result.reciprocal_rank is not None
        ]
        recall_values = [result.recall_at_k for result in results if result.recall_at_k is not None]
        precision_values = [
            result.precision_at_k for result in results if result.precision_at_k is not None
        ]

        return RetrievalEvaluationReport(
            case_count=len(results),
            passed=passed,
            failed=len(results) - passed,
            mean_reciprocal_rank=(
                sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else None
            ),
            mean_recall_at_k=(sum(recall_values) / len(recall_values) if recall_values else None),
            mean_precision_at_k=(
                sum(precision_values) / len(precision_values) if precision_values else None
            ),
            results=results,
        )

    @staticmethod
    def _evaluate_case(
        case: RetrievalEvaluationCase,
        search_result: SemanticItemSearchResult,
    ) -> RetrievalEvaluationCaseResult:
        errors: list[str] = []
        matches = search_result.matches
        returned_ids = sorted(
            {item_id for match in matches for item_id in (match.item_ids or [match.item_id])}
        )

        expected_ids = set(case.expected_item_ids)
        found_expected_ids = expected_ids.intersection(returned_ids)
        recall_at_k: float | None = None
        precision_at_k: float | None = None
        reciprocal_rank: float | None = None

        if expected_ids:
            recall_at_k = len(found_expected_ids) / len(expected_ids)
            relevant_identity_count = sum(
                1
                for match in matches
                if expected_ids.intersection(match.item_ids or [match.item_id])
            )
            precision_at_k = relevant_identity_count / len(matches) if matches else 0.0

            if case.expected_match_mode == "all" and found_expected_ids != expected_ids:
                errors.append(
                    "Not all expected item IDs were retrieved: "
                    f"expected={sorted(expected_ids)}, found={sorted(found_expected_ids)}."
                )
            elif case.expected_match_mode == "any" and not found_expected_ids:
                errors.append(
                    "None of the expected item IDs were retrieved: "
                    f"expected={sorted(expected_ids)}."
                )

            for rank, match in enumerate(matches, start=1):
                if expected_ids.intersection(match.item_ids or [match.item_id]):
                    reciprocal_rank = 1.0 / rank
                    break

        searchable_text = [
            " ".join(
                part
                for part in (
                    match.description,
                    match.normalized_description or "",
                )
                if part
            ).casefold()
            for match in matches
        ]

        missing_terms = [
            term
            for term in case.expected_any_terms
            if not any(term.casefold() in text for text in searchable_text)
        ]
        if case.expected_any_terms and len(missing_terms) == len(case.expected_any_terms):
            errors.append(
                "No expected semantic term appeared in the top results: "
                f"expected_any={case.expected_any_terms}."
            )

        for term in case.forbidden_terms:
            if any(term.casefold() in text for text in searchable_text):
                errors.append(f"Forbidden term appeared in results: {term!r}.")

        forbidden_types = {value.casefold() for value in case.forbidden_parser_item_types}
        returned_forbidden_types = sorted(
            {
                match.parser_item_type
                for match in matches
                if match.parser_item_type and match.parser_item_type.casefold() in forbidden_types
            }
        )
        if returned_forbidden_types:
            errors.append(
                "Forbidden parser item types appeared in results: "
                + ", ".join(returned_forbidden_types)
            )

        return RetrievalEvaluationCaseResult(
            case_id=case.case_id,
            query=case.query,
            passed=not errors,
            top_k=case.top_k,
            returned_item_ids=returned_ids,
            returned_identity_count=len(matches),
            top_descriptions=[match.description for match in matches],
            recall_at_k=recall_at_k,
            precision_at_k=precision_at_k,
            reciprocal_rank=reciprocal_rank,
            errors=errors,
        )
