from __future__ import annotations

from pathlib import Path
from typing import Any

from receipt_intelligence.adapters.storage.sqlite.analytical_query import (
    SQLiteAnalyticalQueryRepository,
)
from receipt_intelligence.observability.ollama import OllamaCallMetrics
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


class FakeAnalyzer:
    def __init__(self, result: QuestionAnalysisResult) -> None:
        self.result = result

    def analyze(self, question: str) -> QuestionAnalysisResult:
        assert question
        return self.result


class FakeRetriever:
    def __init__(self, result: SemanticItemSearchResult) -> None:
        self.result = result
        self.calls = 0

    def search(self, query: str, **_: Any) -> SemanticItemSearchResult:
        self.calls += 1
        assert query == self.result.query
        return self.result


class FakeResolver:
    def __init__(self, result: CandidateResolutionResult) -> None:
        self.result = result
        self.calls = 0

    def resolve(self, *_: Any, **__: Any) -> CandidateResolutionResult:
        self.calls += 1
        return self.result


class FakePlanner:
    def __init__(
        self,
        result: RagSqlPlanResult,
        *,
        repair_result: RagSqlPlanResult | None = None,
    ) -> None:
        self.result = result
        self.repair_result = repair_result
        self.calls = 0
        self.repair_calls = 0
        self.last_validation_error: str | None = None
        self.last_previous_plan: RagSqlPlanResult | None = None

    def plan(self, *_: Any, **__: Any) -> RagSqlPlanResult:
        self.calls += 1
        return self.result

    def repair_after_validation_failure(
        self,
        *_: Any,
        previous_plan: RagSqlPlanResult,
        validation_error: str,
        **__: Any,
    ) -> RagSqlPlanResult:
        self.repair_calls += 1
        self.last_previous_plan = previous_plan
        self.last_validation_error = validation_error
        return self.repair_result or self.result


def _database(tmp_path: Path) -> tuple[ReceiptDatabase, int]:
    db = ReceiptDatabase(tmp_path / "engine.db")
    with db.connect() as connection:
        receipt = connection.execute(
            """
            INSERT INTO receipts(
                job_id, merchant_name, merchant_normalized, receipt_date,
                currency, grand_total, review_status, approved_receipt_path,
                raw_json, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "job-1",
                "Modepark",
                "modepark",
                "2026-07-01",
                "EUR",
                80.74,
                "approved",
                "/approved/job-1.json",
                "{}",
                "2026-07-01T00:00:00+00:00",
                "2026-07-01T00:00:00+00:00",
            ),
        )
        receipt_id = int(receipt.lastrowid)
        item = connection.execute(
            """
            INSERT INTO receipt_items(
                receipt_id, item_index, raw_name, normalized_name,
                parser_item_type, line_total, embedding_text, raw_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (receipt_id, 0, "HS-Halbschuhe", "hs-halbschuhe", "item", 80.74, "shoe", "{}"),
        )
        item_id = int(item.lastrowid)
        connection.commit()
    return db, item_id


def _search_result(
    item_id: int,
    *,
    query: str = "Schuhe",
    description: str = "HS-Halbschuhe",
) -> SemanticItemSearchResult:
    return SemanticItemSearchResult(
        query=query,
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
                description=description,
                normalized_description=description.casefold(),
                similarity=0.8,
                vector_rank=1,
                lexical_rank=1,
                lexical_score=10.0,
                fusion_score=0.04,
            )
        ],
    )


def test_engine_executes_resolved_item_ids_without_dsl(tmp_path: Path) -> None:
    db, item_id = _database(tmp_path)
    analysis = QuestionAnalysisResult(
        status="ready",
        language="de",
        user_goal="Berechne die gesamten Ausgaben für Schuhe.",
        target_entity="spending_amount",
        requested_operation="aggregate_sum",
        requires_product_resolution=True,
        entities=[SemanticEntity(entity_id="e001", search_text="Schuhe")],
        model="test",
        attempts=1,
    )
    resolver_result = CandidateResolutionResult(
        status="resolved",
        semantic_entity="Schuhe",
        candidate_count=1,
        decisions=[],
        selected_candidate_ids=["c001"],
        uncertain_candidate_ids=[],
        rejected_candidate_ids=[],
        selected_item_ids=[item_id],
        model="test",
        attempts=1,
    )
    plan = RagSqlPlanResult(
        status="ready",
        sql=(
            "SELECT ROUND(SUM(line_total), 2) AS value, currency "
            "FROM analytics_purchase_items "
            "WHERE item_id IN (:e001_item_0) GROUP BY currency"
        ),
        parameters={"e001_item_0": item_id},
        result_shape="scalar",
        result_entity="spending_amount",
        display_columns=["value", "currency"],
        answer_instruction="Report spending.",
        model="test",
        attempts=1,
    )
    engine = RagSqlEngine(
        analyzer=FakeAnalyzer(analysis),  # type: ignore[arg-type]
        retriever=FakeRetriever(_search_result(item_id)),
        resolver=FakeResolver(resolver_result),  # type: ignore[arg-type]
        planner=FakePlanner(plan),  # type: ignore[arg-type]
        validator=RagSqlValidator(),
        executor=ReadOnlySqlExecutor(SQLiteAnalyticalQueryRepository(db.db_path)),
    )

    response = engine.execute("Wie viel habe ich für Schuhe ausgegeben?")

    assert response.status == "completed"
    assert response.data is not None
    assert response.data.rows == [{"value": 80.74, "currency": "EUR"}]
    assert response.answer == "Du hast insgesamt 80,74 € für „Schuhe“ ausgegeben."
    assert response.diagnostics["orchestrator"] == "langgraph"
    assert response.diagnostics["graph_version"] == "rag_sql_graph_v2"


def test_engine_preserves_receipt_lookup_semantics(tmp_path: Path) -> None:
    db, item_id = _database(tmp_path)
    analysis = QuestionAnalysisResult(
        status="ready",
        language="de",
        user_goal="Zeige die Quittung oder Quittungen mit Vittel.",
        target_entity="receipt",
        requested_operation="list",
        requires_product_resolution=True,
        entities=[SemanticEntity(entity_id="e001", search_text="Vittel")],
        model="test",
        attempts=1,
    )
    resolver_result = CandidateResolutionResult(
        status="resolved",
        semantic_entity="Vittel",
        candidate_count=1,
        decisions=[],
        selected_candidate_ids=["c001"],
        uncertain_candidate_ids=[],
        rejected_candidate_ids=[],
        selected_item_ids=[item_id],
        model="test",
        attempts=1,
    )
    plan = RagSqlPlanResult(
        status="ready",
        sql=(
            "SELECT DISTINCT R.receipt_id, R.receipt_date, "
            "R.merchant_name AS merchant, R.grand_total, R.currency "
            "FROM analytics_receipts AS R "
            "JOIN analytics_purchase_items AS I "
            "ON I.receipt_id = R.receipt_id "
            "WHERE I.item_id = :e001_item_0 "
            "ORDER BY R.receipt_date DESC, R.receipt_id DESC LIMIT 100"
        ),
        parameters={"e001_item_0": item_id},
        result_shape="rows",
        result_entity="receipt",
        display_columns=[
            "receipt_id",
            "receipt_date",
            "merchant",
            "grand_total",
            "currency",
        ],
        answer_instruction="List the receipts containing Vittel.",
        model="test",
        attempts=1,
    )
    engine = RagSqlEngine(
        analyzer=FakeAnalyzer(analysis),  # type: ignore[arg-type]
        retriever=FakeRetriever(_search_result(item_id, query="Vittel", description="VITTEL 1.5L")),
        resolver=FakeResolver(resolver_result),  # type: ignore[arg-type]
        planner=FakePlanner(plan),  # type: ignore[arg-type]
        validator=RagSqlValidator(),
        executor=ReadOnlySqlExecutor(SQLiteAnalyticalQueryRepository(db.db_path)),
    )

    response = engine.execute("Zeige mir die Quittung mit Vittel.")

    assert response.status == "completed"
    assert response.data is not None
    assert response.data.rows == [
        {
            "receipt_id": 1,
            "receipt_date": "2026-07-01",
            "merchant": "Modepark",
            "grand_total": 80.74,
            "currency": "EUR",
        }
    ]
    assert response.answer == (
        "1 passende Quittung mit „Vittel“ gefunden: "
        "2026-07-01 — Modepark — Gesamtbetrag 80,74 € — Beleg-ID 1."
    )
    assert response.diagnostics["orchestrator"] == "langgraph"
    assert response.diagnostics["graph_version"] == "rag_sql_graph_v2"


def test_engine_stops_on_candidate_clarification_before_sql(tmp_path: Path) -> None:
    db, item_id = _database(tmp_path)
    analysis = QuestionAnalysisResult(
        status="ready",
        language="de",
        user_goal="Berechne die gesamten Ausgaben für Schuhe.",
        target_entity="spending_amount",
        requested_operation="aggregate_sum",
        requires_product_resolution=True,
        entities=[SemanticEntity(entity_id="e001", search_text="Schuhe")],
        model="test",
        attempts=1,
    )
    resolver = FakeResolver(
        CandidateResolutionResult(
            status="needs_clarification",
            semantic_entity="Schuhe",
            candidate_count=1,
            decisions=[],
            selected_candidate_ids=[],
            uncertain_candidate_ids=["c001"],
            rejected_candidate_ids=[],
            selected_item_ids=[],
            uncertain_item_ids=[item_id],
            clarification_question="Zubehör einschließen?",
            model="test",
            attempts=1,
        )
    )
    planner = FakePlanner(
        RagSqlPlanResult(
            status="unsupported",
            reason="must not run",
            model="test",
            attempts=1,
        )
    )
    engine = RagSqlEngine(
        analyzer=FakeAnalyzer(analysis),  # type: ignore[arg-type]
        retriever=FakeRetriever(_search_result(item_id)),
        resolver=resolver,  # type: ignore[arg-type]
        planner=planner,  # type: ignore[arg-type]
        validator=RagSqlValidator(),
        executor=ReadOnlySqlExecutor(SQLiteAnalyticalQueryRepository(db.db_path)),
    )

    response = engine.execute("Schuhe")

    assert response.status == "needs_clarification"
    assert response.clarification_question == "Zubehör einschließen?"
    assert planner.calls == 0


def test_engine_skips_product_retrieval_for_general_spending(tmp_path: Path) -> None:
    db, _ = _database(tmp_path)
    analysis = QuestionAnalysisResult(
        status="ready",
        language="de",
        user_goal="Berechne die gesamten Ausgaben.",
        target_entity="spending_amount",
        requested_operation="aggregate_sum",
        requires_product_resolution=False,
        entities=[],
        model="test",
        attempts=1,
    )
    retriever = FakeRetriever(
        SemanticItemSearchResult(
            query="unused",
            model="embeddinggemma",
            dimension=768,
            total_candidates=0,
            raw_match_count=0,
            limit=1,
            matches=[],
        )
    )
    planner = FakePlanner(
        RagSqlPlanResult(
            status="ready",
            sql=(
                "SELECT ROUND(SUM(grand_total), 2) AS value, currency "
                "FROM analytics_receipts GROUP BY currency"
            ),
            parameters={},
            result_shape="scalar",
            result_entity="spending_amount",
            display_columns=["value", "currency"],
            answer_instruction="Report total receipt spending.",
            model="test",
            attempts=1,
        )
    )
    resolver = FakeResolver(
        CandidateResolutionResult(
            status="not_found",
            semantic_entity="unused",
            candidate_count=0,
            selected_candidate_ids=[],
            uncertain_candidate_ids=[],
            rejected_candidate_ids=[],
            model="test",
            attempts=0,
        )
    )
    engine = RagSqlEngine(
        analyzer=FakeAnalyzer(analysis),  # type: ignore[arg-type]
        retriever=retriever,
        resolver=resolver,  # type: ignore[arg-type]
        planner=planner,  # type: ignore[arg-type]
        validator=RagSqlValidator(),
        executor=ReadOnlySqlExecutor(SQLiteAnalyticalQueryRepository(db.db_path)),
    )

    response = engine.execute("Wie viel habe ich insgesamt ausgegeben?")

    assert response.status == "completed"
    assert response.answer == "Du hast insgesamt 80,74 € ausgegeben."
    assert retriever.calls == 0
    assert resolver.calls == 0
    assert planner.calls == 1


def test_engine_repairs_sql_after_deterministic_validation_failure(
    tmp_path: Path,
) -> None:
    db, item_id = _database(tmp_path)
    analysis = QuestionAnalysisResult(
        status="ready",
        language="de",
        user_goal="Berechne die gesamten Ausgaben für Schuhe.",
        target_entity="spending_amount",
        requested_operation="aggregate_sum",
        requires_product_resolution=True,
        entities=[SemanticEntity(entity_id="e001", search_text="Schuhe")],
        model="test",
        attempts=1,
    )
    resolver_result = CandidateResolutionResult(
        status="resolved",
        semantic_entity="Schuhe",
        candidate_count=1,
        decisions=[],
        selected_candidate_ids=["c001"],
        uncertain_candidate_ids=[],
        rejected_candidate_ids=[],
        selected_item_ids=[item_id],
        model="test",
        attempts=1,
    )
    invalid_plan = RagSqlPlanResult(
        status="ready",
        sql=(
            "SELECT ROUND(SUM(line_total), 2) AS value, currency "
            "FROM analytics_purchase_items "
            "WHERE item_id IN (:e001_item_0) GROUP BY currency"
        ),
        parameters={"e001_item_0": item_id},
        result_shape="grouped_rows",
        result_entity="spending_amount",
        display_columns=["value", "currency"],
        answer_instruction="Report spending.",
        model="test",
        attempts=1,
    )
    repaired_plan = invalid_plan.model_copy(
        update={"sql": f"{invalid_plan.sql} ORDER BY currency LIMIT 100"}
    )
    planner = FakePlanner(invalid_plan, repair_result=repaired_plan)
    engine = RagSqlEngine(
        analyzer=FakeAnalyzer(analysis),  # type: ignore[arg-type]
        retriever=FakeRetriever(_search_result(item_id)),
        resolver=FakeResolver(resolver_result),  # type: ignore[arg-type]
        planner=planner,  # type: ignore[arg-type]
        validator=RagSqlValidator(),
        executor=ReadOnlySqlExecutor(SQLiteAnalyticalQueryRepository(db.db_path)),
        validation_repair_count=1,
    )

    response = engine.execute("Wie viel habe ich für Schuhe ausgegeben?")

    assert response.status == "completed"
    assert response.data is not None
    assert response.data.rows == [{"value": 80.74, "currency": "EUR"}]
    assert response.answer == "Du hast insgesamt 80,74 € für „Schuhe“ ausgegeben."
    assert planner.calls == 1
    assert planner.repair_calls == 1
    assert planner.last_previous_plan == invalid_plan
    assert planner.last_validation_error is not None
    assert "requires a literal LIMIT" in planner.last_validation_error
    assert response.diagnostics["sql_plan"]["sql"].endswith("LIMIT 100")
    assert len(response.diagnostics["sql_plan_attempts"]) == 2
    stages = response.diagnostics["stages"]
    assert any(
        stage["name"] == "validate_sql_attempt_1" and stage["status"] == "error" for stage in stages
    )
    assert any(stage["name"] == "repair_sql_attempt_1" for stage in stages)
    assert any(
        stage["name"] == "validate_sql_attempt_2" and stage["status"] == "done" for stage in stages
    )
    assert response.diagnostics["orchestrator"] == "langgraph"
    assert response.diagnostics["graph_version"] == "rag_sql_graph_v2"


def test_engine_returns_validation_error_when_repaired_sql_is_still_invalid(
    tmp_path: Path,
) -> None:
    db, _ = _database(tmp_path)
    analysis = QuestionAnalysisResult(
        status="ready",
        language="de",
        user_goal="Berechne die gesamten Ausgaben.",
        target_entity="spending_amount",
        requested_operation="aggregate_sum",
        requires_product_resolution=False,
        entities=[],
        model="test",
        attempts=1,
    )
    invalid_plan = RagSqlPlanResult(
        status="ready",
        sql=(
            "SELECT ROUND(SUM(grand_total), 2) AS value, currency "
            "FROM analytics_receipts GROUP BY currency"
        ),
        parameters={},
        result_shape="grouped_rows",
        result_entity="spending_amount",
        display_columns=["value", "currency"],
        answer_instruction="Report spending.",
        model="test",
        attempts=1,
    )
    planner = FakePlanner(invalid_plan, repair_result=invalid_plan)
    retriever = FakeRetriever(
        SemanticItemSearchResult(
            query="unused",
            model="embeddinggemma",
            dimension=768,
            total_candidates=0,
            raw_match_count=0,
            limit=1,
            matches=[],
        )
    )
    resolver = FakeResolver(
        CandidateResolutionResult(
            status="not_found",
            semantic_entity="unused",
            candidate_count=0,
            selected_candidate_ids=[],
            uncertain_candidate_ids=[],
            rejected_candidate_ids=[],
            model="test",
            attempts=0,
        )
    )
    engine = RagSqlEngine(
        analyzer=FakeAnalyzer(analysis),  # type: ignore[arg-type]
        retriever=retriever,
        resolver=resolver,  # type: ignore[arg-type]
        planner=planner,  # type: ignore[arg-type]
        validator=RagSqlValidator(),
        executor=ReadOnlySqlExecutor(SQLiteAnalyticalQueryRepository(db.db_path)),
        validation_repair_count=1,
    )

    response = engine.execute("Wie viel habe ich insgesamt ausgegeben?")

    assert response.status == "error"
    assert response.error_code == "sql_validation_failed"
    assert response.error is not None
    assert "requires a literal LIMIT" in response.error
    assert planner.calls == 1
    assert planner.repair_calls == 1
    assert response.diagnostics["orchestrator"] == "langgraph"
    assert response.diagnostics["graph_version"] == "rag_sql_graph_v2"


def test_engine_can_disable_validation_repair(tmp_path: Path) -> None:
    db, _ = _database(tmp_path)
    analysis = QuestionAnalysisResult(
        status="ready",
        language="de",
        user_goal="Berechne die gesamten Ausgaben.",
        target_entity="spending_amount",
        requested_operation="aggregate_sum",
        requires_product_resolution=False,
        entities=[],
        model="test",
        attempts=1,
    )
    invalid_plan = RagSqlPlanResult(
        status="ready",
        sql=(
            "SELECT ROUND(SUM(grand_total), 2) AS value, currency "
            "FROM analytics_receipts GROUP BY currency"
        ),
        parameters={},
        result_shape="grouped_rows",
        result_entity="spending_amount",
        display_columns=["value", "currency"],
        answer_instruction="Report spending.",
        model="test",
        attempts=1,
    )
    planner = FakePlanner(invalid_plan)
    engine = RagSqlEngine(
        analyzer=FakeAnalyzer(analysis),  # type: ignore[arg-type]
        retriever=FakeRetriever(
            SemanticItemSearchResult(
                query="unused",
                model="embeddinggemma",
                dimension=768,
                total_candidates=0,
                raw_match_count=0,
                limit=1,
                matches=[],
            )
        ),
        resolver=FakeResolver(  # type: ignore[arg-type]
            CandidateResolutionResult(
                status="not_found",
                semantic_entity="unused",
                candidate_count=0,
                selected_candidate_ids=[],
                uncertain_candidate_ids=[],
                rejected_candidate_ids=[],
                model="test",
                attempts=0,
            )
        ),
        planner=planner,  # type: ignore[arg-type]
        validator=RagSqlValidator(),
        executor=ReadOnlySqlExecutor(SQLiteAnalyticalQueryRepository(db.db_path)),
        validation_repair_count=0,
    )

    response = engine.execute("Wie viel habe ich insgesamt ausgegeben?")

    assert response.status == "error"
    assert response.error_code == "sql_validation_failed"
    assert planner.repair_calls == 0


def test_engine_exposes_ollama_stage_metrics_and_summary(tmp_path: Path) -> None:
    db, _ = _database(tmp_path)
    analysis_call = OllamaCallMetrics(
        endpoint="generate",
        model="gemma4",
        request_duration_ms=1000.0,
        total_duration_ns=900_000_000,
        load_duration_ns=300_000_000,
        prompt_eval_count=20,
        prompt_eval_duration_ns=200_000_000,
        eval_count=10,
        eval_duration_ns=300_000_000,
    )
    planner_call = OllamaCallMetrics(
        endpoint="generate",
        model="gemma4",
        request_duration_ms=800.0,
        total_duration_ns=700_000_000,
        load_duration_ns=0,
        prompt_eval_count=30,
        prompt_eval_duration_ns=300_000_000,
        eval_count=8,
        eval_duration_ns=200_000_000,
    )
    analysis = QuestionAnalysisResult(
        status="ready",
        language="de",
        user_goal="Berechne die gesamten Ausgaben.",
        target_entity="spending_amount",
        requested_operation="aggregate_sum",
        requires_product_resolution=False,
        entities=[],
        model="gemma4",
        attempts=1,
        ollama_calls=[analysis_call],
    )
    plan = RagSqlPlanResult(
        status="ready",
        sql=(
            "SELECT currency, ROUND(SUM(grand_total), 2) AS value "
            "FROM analytics_receipts GROUP BY currency ORDER BY currency LIMIT 100"
        ),
        parameters={},
        result_shape="grouped_rows",
        result_entity="spending_amount",
        display_columns=["currency", "value"],
        answer_instruction="Report total receipt spending.",
        model="gemma4",
        attempts=1,
        ollama_calls=[planner_call],
    )
    retriever = FakeRetriever(
        SemanticItemSearchResult(
            query="unused",
            model="embeddinggemma",
            dimension=768,
            total_candidates=0,
            raw_match_count=0,
            limit=1,
            matches=[],
        )
    )
    resolver = FakeResolver(
        CandidateResolutionResult(
            status="not_found",
            semantic_entity="unused",
            candidate_count=0,
            selected_candidate_ids=[],
            uncertain_candidate_ids=[],
            rejected_candidate_ids=[],
            model="gemma4",
            attempts=0,
        )
    )
    engine = RagSqlEngine(
        analyzer=FakeAnalyzer(analysis),  # type: ignore[arg-type]
        retriever=retriever,
        resolver=resolver,  # type: ignore[arg-type]
        planner=FakePlanner(plan),  # type: ignore[arg-type]
        validator=RagSqlValidator(),
        executor=ReadOnlySqlExecutor(SQLiteAnalyticalQueryRepository(db.db_path)),
    )

    response = engine.execute("Wie viel habe ich insgesamt ausgegeben?")

    assert response.status == "completed"
    stages = response.diagnostics["stages"]
    analyze_stage = next(stage for stage in stages if stage["name"] == "analyze_question")
    planner_stage = next(stage for stage in stages if stage["name"] == "generate_sql")
    assert analyze_stage["ollama_calls"][0]["load_duration_ms"] == 300.0
    assert planner_stage["ollama_calls"][0]["load_duration_ms"] == 0.0
    summary = response.diagnostics["ollama_summary"]
    assert summary["call_count"] == 2
    assert summary["total_load_duration_ms"] == 300.0
    assert summary["total_prompt_eval_duration_ms"] == 500.0
    assert summary["total_generation_duration_ms"] == 500.0


def test_engine_returns_insufficient_info_for_unsupported_brand_metadata(
    tmp_path: Path,
) -> None:
    db, item_id = _database(tmp_path)
    analysis = QuestionAnalysisResult(
        status="ready",
        language="en",
        user_goal="Identify the brand of the resolved Vittel item.",
        target_entity="product_brand",
        requested_operation="identify_brand",
        requires_product_resolution=True,
        entities=[SemanticEntity(entity_id="e001", search_text="Vittel")],
        model="test",
        attempts=1,
    )
    resolver_result = CandidateResolutionResult(
        status="resolved",
        semantic_entity="Vittel",
        candidate_count=1,
        decisions=[],
        selected_candidate_ids=["c001"],
        uncertain_candidate_ids=[],
        rejected_candidate_ids=[],
        selected_item_ids=[item_id],
        model="test",
        attempts=1,
    )
    plan = RagSqlPlanResult(
        status="ready",
        sql=(
            "SELECT item_id, description, normalized_name, semantic_description, "
            "category, category_reason FROM analytics_purchase_items "
            "WHERE item_id = :e001_item_0 ORDER BY item_id LIMIT 100"
        ),
        parameters={"e001_item_0": item_id},
        result_shape="rows",
        result_entity="product_brand",
        display_columns=[
            "item_id",
            "description",
            "normalized_name",
            "semantic_description",
            "category",
            "category_reason",
        ],
        answer_instruction="Use only reviewed metadata.",
        model="test",
        attempts=1,
    )
    engine = RagSqlEngine(
        analyzer=FakeAnalyzer(analysis),  # type: ignore[arg-type]
        retriever=FakeRetriever(_search_result(item_id, query="Vittel", description="VITTEL 1.5L")),
        resolver=FakeResolver(resolver_result),  # type: ignore[arg-type]
        planner=FakePlanner(plan),  # type: ignore[arg-type]
        validator=RagSqlValidator(),
        executor=ReadOnlySqlExecutor(SQLiteAnalyticalQueryRepository(db.db_path)),
    )

    response = engine.execute("Which brand is Vittel?")

    assert response.status == "insufficient_info"
    assert response.data is not None and response.data.row_count == 1
    assert "enough product information" in response.answer
