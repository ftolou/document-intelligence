from __future__ import annotations

from receipt_intelligence.rag_sql.application import ReceiptQueryService
from receipt_intelligence.rag_sql.models import RagSqlResponse


class _Runtime:
    def __init__(self) -> None:
        self.close_count = 0

    def execute(self, question: str) -> RagSqlResponse:
        return RagSqlResponse(question=question, status="completed", answer="ok")

    def close(self) -> None:
        self.close_count += 1


def test_query_service_releases_runtime() -> None:
    runtime = _Runtime()
    service = ReceiptQueryService(runtime=runtime)  # type: ignore[arg-type]

    service.close()

    assert runtime.close_count == 1
