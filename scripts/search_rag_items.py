"""Search approved receipt items using the semantic embedding index."""

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
from receipt_intelligence.adapters.embeddings import (  # noqa: E402
    EmbeddingProviderConfig,
    build_embedding_gateway,
)
from receipt_intelligence.adapters.storage.sqlite.semantic_search import (  # noqa: E402
    SQLiteSemanticSearchRepository,
)
from receipt_intelligence.rag.item_retriever import ItemSemanticRetriever  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query")
    parser.add_argument("--db", type=Path, default=settings.RECEIPT_DB_PATH)
    parser.add_argument(
        "--provider",
        choices=("ollama", "openai"),
        default=settings.RAG_EMBEDDING_PROVIDER,
    )
    parser.add_argument("--model", default=settings.RAG_EMBEDDING_MODEL)
    parser.add_argument(
        "--base-url",
        "--ollama-url",
        dest="base_url",
        default=settings.RAG_EMBEDDING_BASE_URL,
    )
    parser.add_argument(
        "--dimensions",
        type=int,
        default=settings.RAG_EMBEDDING_DIMENSIONS,
    )
    parser.add_argument("--limit", type=int, default=settings.RAG_RETRIEVAL_DEFAULT_LIMIT)
    parser.add_argument(
        "--minimum-score",
        type=float,
        default=settings.RAG_RETRIEVAL_MINIMUM_SCORE,
    )
    parser.add_argument("--merchant")
    parser.add_argument("--category")
    parser.add_argument(
        "--no-deduplicate",
        action="store_true",
        help="Return individual purchase occurrences instead of product identities.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=settings.RAG_EMBEDDING_TIMEOUT_SECONDS,
    )
    parser.add_argument("--keep-alive", default=settings.RAG_EMBEDDING_KEEP_ALIVE)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    config = EmbeddingProviderConfig(
        provider=args.provider,
        model=args.model,
        base_url=args.base_url or None,
        api_key=settings.OPENAI_API_KEY if args.provider == "openai" else None,
        dimensions=args.dimensions,
        timeout_seconds=args.timeout_seconds,
        keep_alive=args.keep_alive if args.provider == "ollama" else None,
    )
    with build_embedding_gateway(config) as client:
        result = ItemSemanticRetriever(
            repository=SQLiteSemanticSearchRepository(args.db),
            embedding_client=client,
            maximum_limit=settings.RAG_RETRIEVAL_MAX_LIMIT,
            deduplicate=settings.RAG_RETRIEVAL_DEDUPLICATE,
            rrf_k=settings.RAG_RETRIEVAL_RRF_K,
            vector_weight=settings.RAG_RETRIEVAL_VECTOR_WEIGHT,
            lexical_weight=settings.RAG_RETRIEVAL_LEXICAL_WEIGHT,
        ).search(
            args.query,
            limit=args.limit,
            minimum_score=args.minimum_score,
            merchant=args.merchant,
            category=args.category,
            deduplicate=not args.no_deduplicate,
        )

    print(json.dumps(result.model_dump(mode="json"), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
