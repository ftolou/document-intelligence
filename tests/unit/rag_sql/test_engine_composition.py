from __future__ import annotations

import pytest

from receipt_intelligence.rag_sql.engine import RagSqlEngine
from receipt_intelligence.rag_sql.models import RagSqlResponse
from receipt_intelligence.rag_sql.orchestration.contracts import RagSqlComponents
from receipt_intelligence.rag_sql.planner import RagSqlPlanner, RagSqlPlannerConfig
from receipt_intelligence.rag_sql.schema_catalog import schema_catalog_for_dialect
from receipt_intelligence.rag_sql.validator import RagSqlValidator, SqlValidatorConfig


class _FakeOrchestrator:
    name = "fake"
    version = "fake-v1"

    def __init__(self) -> None:
        self.questions: list[str] = []

    def execute(self, question: str) -> RagSqlResponse:
        self.questions.append(question)
        return RagSqlResponse(question=question, status="completed", answer="done")


class _FakeFactory:
    def __init__(self, orchestrator: _FakeOrchestrator) -> None:
        self.orchestrator = orchestrator
        self.components: RagSqlComponents | None = None
        self.build_count = 0

    def build(self, components, *, graph_config):
        self.components = components
        self.build_count += 1
        return self.orchestrator


def _planner(sql_dialect: str = "sqlite") -> RagSqlPlanner:
    return RagSqlPlanner(
        RagSqlPlannerConfig(sql_dialect=sql_dialect),
        schema_catalog=schema_catalog_for_dialect(sql_dialect),
    )


def _validator(sql_dialect: str = "sqlite") -> RagSqlValidator:
    return RagSqlValidator(SqlValidatorConfig(sql_dialect=sql_dialect))


def _engine(
    *,
    planner: RagSqlPlanner,
    validator: RagSqlValidator,
    factory: _FakeFactory,
) -> RagSqlEngine:
    return RagSqlEngine(
        analyzer=object(),  # type: ignore[arg-type]
        retriever=object(),  # type: ignore[arg-type]
        resolver=object(),  # type: ignore[arg-type]
        planner=planner,
        validator=validator,
        executor=object(),  # type: ignore[arg-type]
        orchestrator_factory=factory,
    )


def test_engine_uses_injected_orchestrator_without_importing_langgraph() -> None:
    orchestrator = _FakeOrchestrator()
    factory = _FakeFactory(orchestrator)

    engine = _engine(planner=_planner(), validator=_validator(), factory=factory)

    assert factory.build_count == 1
    assert isinstance(factory.components, RagSqlComponents)
    assert engine.orchestrator_name == "fake"
    assert engine.orchestrator_version == "fake-v1"
    assert engine.execute("  what   did I buy? ").answer == "done"
    assert orchestrator.questions == ["what did I buy?"]


def test_engine_composes_matching_postgresql_planner_and_validator() -> None:
    orchestrator = _FakeOrchestrator()
    factory = _FakeFactory(orchestrator)

    _engine(
        planner=_planner("postgresql"),
        validator=_validator("postgresql"),
        factory=factory,
    )

    assert factory.build_count == 1
    assert factory.components is not None
    assert factory.components.planner.sql_dialect.name == "postgresql"
    assert factory.components.validator.sql_dialect.name == "postgresql"


def test_engine_rejects_planner_validator_dialect_mismatch_before_composition() -> None:
    orchestrator = _FakeOrchestrator()
    factory = _FakeFactory(orchestrator)

    with pytest.raises(ValueError, match="planner and validator SQL dialects must match"):
        _engine(
            planner=_planner("postgresql"),
            validator=_validator("sqlite"),
            factory=factory,
        )

    assert factory.build_count == 0
