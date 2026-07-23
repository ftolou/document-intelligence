from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from receipt_intelligence.rag.candidate_models import CandidateResolutionResult
from receipt_intelligence.rag.models import SemanticItemMatch, SemanticItemSearchResult
from receipt_intelligence.rag_sql.engine import RagSqlEngine
from receipt_intelligence.rag_sql.executor import ReadOnlySqlExecutor
from receipt_intelligence.rag_sql.models import (
    QuestionAnalysisResult,
    RagSqlPlanResult,
    SemanticEntity,
)
from receipt_intelligence.rag_sql.validator import RagSqlValidator
from receipt_intelligence.storage.receipt_db import ReceiptDatabase


@dataclass
class _Analyzer:
    result: QuestionAnalysisResult

    def analyze(self, _: str) -> QuestionAnalysisResult:
        return self.result


@dataclass
class _Planner:
    plan_result: RagSqlPlanResult

    def plan(self, *_: Any, protected_parameters: dict[str, int], **__: Any) -> RagSqlPlanResult:
        if protected_parameters:
            return self.plan_result.model_copy(update={"parameters": dict(protected_parameters)})
        return self.plan_result

    def repair_after_validation_failure(self, *_: Any, **__: Any) -> RagSqlPlanResult:
        raise AssertionError("Regression plans must validate without repair.")


@dataclass
class _Retriever:
    result: SemanticItemSearchResult | None = None
    calls: int = 0

    def search(self, query: str, **_: Any) -> SemanticItemSearchResult:
        self.calls += 1
        if self.result is None:
            raise AssertionError("Retrieval was not expected for this case.")
        assert query == self.result.query
        return self.result


@dataclass
class _Resolver:
    item_id: int | None = None
    calls: int = 0

    def resolve(self, semantic_entity: str, *_: Any, **__: Any) -> CandidateResolutionResult:
        self.calls += 1
        if self.item_id is None:
            raise AssertionError("Resolution was not expected for this case.")
        return CandidateResolutionResult(
            status="resolved",
            semantic_entity=semantic_entity,
            candidate_count=1,
            decisions=[],
            selected_candidate_ids=["c001"],
            uncertain_candidate_ids=[],
            rejected_candidate_ids=[],
            selected_item_ids=[self.item_id],
            model="test",
            attempts=1,
        )


def _database(tmp_path: Path) -> ReceiptDatabase:
    db = ReceiptDatabase(tmp_path / "query-corpus.db")
    receipts = [
        {
            "job_id": "june_dm",
            "merchant": {"name": "dm-drogerie markt"},
            "date": "2026-06-12",
            "currency": "EUR",
            "totals": {"grand_total": 10.0, "paid_total": 10.0},
            "items": [
                {
                    "description": "HEAD&SHOULDERS CLASSIC",
                    "normalized_name": "head shoulders classic",
                    "category": "item",
                    "category_key": "personal_care/shampoo",
                    "line_total": 3.95,
                },
                {
                    "description": "WASSER",
                    "normalized_name": "wasser",
                    "category": "item",
                    "category_key": "groceries_beverages",
                    "line_total": 0.79,
                },
            ],
            "human_review": {"status": "approved"},
        },
        {
            "job_id": "june_rewe",
            "merchant": {"name": "REWE"},
            "date": "2026-06-18",
            "currency": "EUR",
            "totals": {"grand_total": 20.0, "paid_total": 20.0},
            "items": [
                {"description": "MILCH", "category": "item", "line_total": 2.0},
                {"description": "SCHUHE", "category": "item", "line_total": 49.0},
            ],
            "human_review": {"status": "approved"},
        },
        {
            "job_id": "july_rewe",
            "merchant": {"name": "REWE"},
            "date": "2026-07-05",
            "currency": "EUR",
            "totals": {"grand_total": 45.0, "paid_total": 45.0},
            "items": [
                {"description": "BROT", "category": "item", "line_total": 3.0},
                {"description": "WASSER", "category": "item", "line_total": 0.99},
            ],
            "human_review": {"status": "approved"},
        },
    ]
    for receipt in receipts:
        db.import_receipt(job_id=receipt["job_id"], receipt=receipt)
    return db


def _analysis(
    *,
    language: str,
    operation: str,
    target: str,
    entity: str | None = None,
) -> QuestionAnalysisResult:
    entities = [SemanticEntity(entity_id="e001", search_text=entity)] if entity else []
    return QuestionAnalysisResult(
        status="ready",
        language=language,
        user_goal="Regression query",
        target_entity=target,
        requested_operation=operation,
        requires_product_resolution=bool(entities),
        entities=entities,
        model="test",
        attempts=1,
    )


def _plan(
    sql: str,
    *,
    parameters: dict[str, Any] | None = None,
    shape: str = "scalar",
    entity: str = "spending_amount",
    columns: list[str] | None = None,
) -> RagSqlPlanResult:
    return RagSqlPlanResult(
        status="ready",
        sql=sql,
        parameters=parameters or {},
        result_shape=shape,
        result_entity=entity,
        display_columns=columns or ["value", "currency"],
        answer_instruction="Return the deterministic result.",
        model="test",
        attempts=1,
    )


@pytest.mark.regression
def test_rag_sql_langgraph_query_corpus(tmp_path: Path) -> None:
    db = _database(tmp_path)
    with db.connect() as connection:
        shampoo_id = int(
            connection.execute(
                "SELECT id FROM receipt_items WHERE lower(raw_name) = 'head shoulders classic'"
            ).fetchone()[0]
        )

    cases = [
        {
            "id": "merchant_month_sum",
            "question": "How much did I spend at REWE in 2026-06?",
            "analysis": _analysis(
                language="en", operation="aggregate_sum", target="spending_amount"
            ),
            "plan": _plan(
                "SELECT ROUND(SUM(grand_total), 2) AS value, currency "
                "FROM analytics_receipts WHERE lower(merchant_name) = :merchant "
                "AND receipt_month = :month GROUP BY currency",
                parameters={"merchant": "rewe", "month": "2026-06"},
            ),
            "expected_rows": [{"value": 20.0, "currency": "EUR"}],
        },
        {
            "id": "semantic_item_sum",
            "question": "How much did I spend on shampoo?",
            "analysis": _analysis(
                language="en",
                operation="aggregate_sum",
                target="spending_amount",
                entity="shampoo",
            ),
            "plan": _plan(
                "SELECT ROUND(SUM(line_total), 2) AS value, currency "
                "FROM analytics_purchase_items WHERE item_id = :e001_item_0 "
                "GROUP BY currency"
            ),
            "semantic_item_id": shampoo_id,
            "expected_rows": [{"value": 3.95, "currency": "EUR"}],
        },
        {
            "id": "merchant_grouping",
            "question": "Break down my spending by merchant in 2026.",
            "analysis": _analysis(language="en", operation="group_sum", target="merchant_spending"),
            "plan": _plan(
                "SELECT lower(merchant_name) AS merchant, "
                "ROUND(SUM(grand_total), 2) AS value, currency "
                "FROM analytics_receipts WHERE substr(receipt_date, 1, 4) = :year "
                "GROUP BY lower(merchant_name), currency ORDER BY merchant LIMIT 100",
                parameters={"year": "2026"},
                shape="grouped_rows",
                entity="merchant_spending",
                columns=["merchant", "value", "currency"],
            ),
            "expected_rows": [
                {"merchant": "dm-drogerie markt", "value": 10.0, "currency": "EUR"},
                {"merchant": "rewe", "value": 65.0, "currency": "EUR"},
            ],
        },
        {
            "id": "receipt_count",
            "question": "How many receipts do I have from REWE?",
            "analysis": _analysis(language="en", operation="count", target="receipt_count"),
            "plan": _plan(
                "SELECT COUNT(*) AS value FROM analytics_receipts "
                "WHERE lower(merchant_name) = :merchant",
                parameters={"merchant": "rewe"},
                entity="receipt_count",
                columns=["value"],
            ),
            "expected_rows": [{"value": 2}],
        },
        {
            "id": "cheapest_unique_item",
            "question": "Show me the cheapest distinct item.",
            "analysis": _analysis(language="en", operation="minimum", target="product"),
            "plan": _plan(
                "SELECT description, MIN(line_total) AS line_total, currency "
                "FROM analytics_purchase_items GROUP BY lower(description), currency "
                "ORDER BY line_total ASC LIMIT 1",
                shape="row",
                entity="product",
                columns=["description", "line_total", "currency"],
            ),
            "expected_rows": [{"description": "wasser", "line_total": 0.79, "currency": "EUR"}],
        },
        {
            "id": "german_merchant_sum",
            "question": "Wie viel habe ich bei REWE ausgegeben?",
            "analysis": _analysis(
                language="de", operation="aggregate_sum", target="spending_amount"
            ),
            "plan": _plan(
                "SELECT ROUND(SUM(grand_total), 2) AS value, currency "
                "FROM analytics_receipts WHERE lower(merchant_name) = :merchant "
                "GROUP BY currency",
                parameters={"merchant": "rewe"},
            ),
            "expected_rows": [{"value": 65.0, "currency": "EUR"}],
        },
    ]

    for case in cases:
        item_id = case.get("semantic_item_id")
        search_result = None
        if item_id:
            search_result = SemanticItemSearchResult(
                query="shampoo",
                model="embeddinggemma",
                dimension=768,
                total_candidates=1,
                raw_match_count=1,
                limit=12,
                matches=[
                    SemanticItemMatch(
                        rank=1,
                        item_id=item_id,
                        item_ids=[item_id],
                        occurrence_count=1,
                        receipt_id=1,
                        description="HEAD&SHOULDERS CLASSIC",
                        normalized_description="head shoulders classic",
                        similarity=0.9,
                        vector_rank=1,
                        lexical_rank=1,
                        lexical_score=10.0,
                        fusion_score=0.04,
                    )
                ],
            )
        retriever = _Retriever(search_result)
        resolver = _Resolver(item_id)
        engine = RagSqlEngine(
            analyzer=_Analyzer(case["analysis"]),  # type: ignore[arg-type]
            retriever=retriever,
            resolver=resolver,  # type: ignore[arg-type]
            planner=_Planner(case["plan"]),  # type: ignore[arg-type]
            validator=RagSqlValidator(),
            executor=ReadOnlySqlExecutor(db.db_path),
        )

        response = engine.execute(case["question"])

        assert response.status == "completed", case["id"]
        assert response.data is not None, case["id"]
        assert response.data.rows == case["expected_rows"], case["id"]
        assert response.diagnostics["orchestrator"] == "langgraph", case["id"]
        assert response.diagnostics["graph_version"] == "rag_sql_graph_v2", case["id"]
        assert response.diagnostics["orchestrator"] == "langgraph", case["id"]
        assert retriever.calls == (1 if item_id else 0), case["id"]
        assert resolver.calls == (1 if item_id else 0), case["id"]
