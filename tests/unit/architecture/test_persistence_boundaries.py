from __future__ import annotations

from dataclasses import dataclass, field

from receipt_intelligence.rag.item_indexer import ItemEmbeddingIndexer
from receipt_intelligence.rag.item_retriever import ItemSemanticRetriever
from receipt_intelligence.rag.models import EmbeddingBatchResult
from receipt_intelligence.rag.ports import (
    IndexableItemSource,
    SemanticIndexState,
    SemanticSearchCandidate,
    StoredItemEmbedding,
)
from receipt_intelligence.rag_sql.executor import ReadOnlySqlExecutor
from receipt_intelligence.rag_sql.models import SqlExecutionResult, ValidatedSqlPlan


class _EmbeddingClient:
    model = "test-model"

    def embed(self, texts: list[str]) -> EmbeddingBatchResult:
        return EmbeddingBatchResult(
            model=self.model,
            vectors=[[1.0, 0.0] for _ in texts],
            dimension=2,
        )


@dataclass
class _IndexRepository:
    stored: list[StoredItemEmbedding] = field(default_factory=list)
    states: list[SemanticIndexState] = field(default_factory=list)

    def load_indexable_items(self, *, approved_only: bool, item_ids=None):
        assert approved_only is True
        return [
            IndexableItemSource(
                item_id=1,
                receipt_id=2,
                description="VITTEL",
                description_normalized="vittel",
                category="beverages",
                category_group=None,
                category_key=None,
                category_reason="Mineral water",
                semantic_description=None,
                item_raw_json={},
                merchant="REWE",
                parser_item_type="item",
            )
        ]

    def prune_embeddings(self, **_kwargs) -> int:
        return 0

    def existing_hashes(self, **_kwargs) -> dict[int, str]:
        return {}

    def known_dimension(self, **_kwargs) -> int | None:
        return None

    def store_embeddings(self, records) -> None:
        self.stored.extend(records)

    def save_state(self, state: SemanticIndexState) -> None:
        self.states.append(state)


@dataclass
class _SearchRepository:
    def load_candidates(self, **_kwargs):
        return [
            SemanticSearchCandidate(
                item_id=1,
                embedding_dimension=2,
                vector=(1.0, 0.0),
                receipt_id=2,
                description="VITTEL",
                normalized_description="vittel",
                category="beverages",
                item_raw_json={"category_reason": "Mineral water"},
                parser_item_type="item",
                line_total=1.29,
                unit_price=1.29,
                merchant="REWE",
                receipt_date="2026-07-01",
                currency="EUR",
            )
        ]

    def load_fts_ranks(self, **_kwargs):
        return {1: 1}


@dataclass
class _AnalyticalRepository:
    calls: list[tuple[int, float, int]] = field(default_factory=list)

    def execute(
        self,
        plan: ValidatedSqlPlan,
        *,
        maximum_rows: int,
        timeout_seconds: float,
        progress_opcodes: int,
    ) -> SqlExecutionResult:
        assert plan.sql.startswith("SELECT")
        self.calls.append((maximum_rows, timeout_seconds, progress_opcodes))
        return SqlExecutionResult(
            columns=["value"],
            rows=[{"value": 1.29}],
            row_count=1,
            truncated=False,
            duration_ms=0.1,
        )


def test_indexer_depends_only_on_semantic_index_repository() -> None:
    repository = _IndexRepository()
    report = ItemEmbeddingIndexer(
        repository=repository,
        embedding_client=_EmbeddingClient(),
    ).rebuild()

    assert report.embedded == 1
    assert repository.stored[0].item_id == 1
    assert repository.states[-1].indexed_count == 1


def test_retriever_depends_only_on_semantic_search_repository() -> None:
    result = ItemSemanticRetriever(
        repository=_SearchRepository(),
        embedding_client=_EmbeddingClient(),
    ).search("Vittel")

    assert result.matches[0].description == "VITTEL"
    assert result.matches[0].line_total == 1.29


def test_sql_executor_delegates_to_analytical_repository() -> None:
    repository = _AnalyticalRepository()
    executor = ReadOnlySqlExecutor(repository)
    plan = ValidatedSqlPlan(
        sql="SELECT :value AS value",
        parameters={"value": 1.29},
        result_shape="scalar",
        result_entity="value",
        display_columns=["value"],
        answer_instruction="Report the value.",
        referenced_objects=[],
        referenced_functions=[],
        placeholder_names=["value"],
    )

    result = executor.execute(plan)

    assert result.rows == [{"value": 1.29}]
    assert repository.calls == [(100, 5.0, 1000)]
