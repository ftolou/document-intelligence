"""Evaluate semantic item retrieval against a user-maintained JSON corpus."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from receipt_intelligence import settings  # noqa: E402
from receipt_intelligence.adapters.storage.sqlite.semantic_search import (  # noqa: E402
    SQLiteSemanticSearchRepository,
)
from receipt_intelligence.rag.embedding_client import OllamaEmbeddingClient  # noqa: E402
from receipt_intelligence.rag.item_retriever import ItemSemanticRetriever  # noqa: E402
from receipt_intelligence.rag.retrieval_evaluator import (  # noqa: E402
    ItemRetrievalEvaluator,
    load_evaluation_cases,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--db", type=Path, default=settings.RECEIPT_DB_PATH)
    parser.add_argument("--model", default=settings.RAG_EMBEDDING_MODEL)
    parser.add_argument("--ollama-url", default=settings.OLLAMA_URL)
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=settings.RAG_EMBEDDING_TIMEOUT_SECONDS,
    )
    parser.add_argument("--keep-alive", default=settings.RAG_EMBEDDING_KEEP_ALIVE)
    parser.add_argument(
        "--fail-on-miss",
        action="store_true",
        help="Return exit code 1 when at least one retrieval case fails.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    cases = load_evaluation_cases(args.cases)

    with OllamaEmbeddingClient(
        base_url=args.ollama_url,
        model=args.model,
        timeout_seconds=args.timeout_seconds,
        keep_alive=args.keep_alive or None,
    ) as client:
        retriever = ItemSemanticRetriever(
            repository=SQLiteSemanticSearchRepository(args.db),
            embedding_client=client,
            maximum_limit=settings.RAG_RETRIEVAL_MAX_LIMIT,
            deduplicate=settings.RAG_RETRIEVAL_DEDUPLICATE,
            rrf_k=settings.RAG_RETRIEVAL_RRF_K,
            vector_weight=settings.RAG_RETRIEVAL_VECTOR_WEIGHT,
            lexical_weight=settings.RAG_RETRIEVAL_LEXICAL_WEIGHT,
        )
        report = ItemRetrievalEvaluator(retriever).evaluate(cases)

    print(json.dumps(report.model_dump(mode="json"), indent=2, ensure_ascii=False))
    return 1 if args.fail_on_miss and report.failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
