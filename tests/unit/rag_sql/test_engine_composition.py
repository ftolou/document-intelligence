from __future__ import annotations

from receipt_intelligence.rag_sql.engine import RagSqlEngine
from receipt_intelligence.rag_sql.models import RagSqlResponse
from receipt_intelligence.rag_sql.orchestration.contracts import RagSqlComponents


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


def test_engine_uses_injected_orchestrator_without_importing_langgraph() -> None:
    orchestrator = _FakeOrchestrator()
    factory = _FakeFactory(orchestrator)

    engine = RagSqlEngine(
        analyzer=object(),  # type: ignore[arg-type]
        retriever=object(),  # type: ignore[arg-type]
        resolver=object(),  # type: ignore[arg-type]
        planner=object(),  # type: ignore[arg-type]
        validator=object(),  # type: ignore[arg-type]
        executor=object(),  # type: ignore[arg-type]
        orchestrator_factory=factory,
    )

    assert factory.build_count == 1
    assert isinstance(factory.components, RagSqlComponents)
    assert engine.orchestrator_name == "fake"
    assert engine.orchestrator_version == "fake-v1"
    assert engine.execute("  what   did I buy? ").answer == "done"
    assert orchestrator.questions == ["what did I buy?"]
