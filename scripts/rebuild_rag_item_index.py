"""Build or incrementally refresh approved receipt-item embeddings."""

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
from receipt_intelligence.adapters.storage.sqlite.semantic_index import (  # noqa: E402
    SQLiteSemanticIndexRepository,
)
from receipt_intelligence.rag.item_indexer import ItemEmbeddingIndexer  # noqa: E402
from receipt_intelligence.storage.bootstrap import initialize_database  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
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
    parser.add_argument("--batch-size", type=int, default=settings.RAG_EMBEDDING_BATCH_SIZE)
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=settings.RAG_EMBEDDING_TIMEOUT_SECONDS,
    )
    parser.add_argument("--keep-alive", default=settings.RAG_EMBEDDING_KEEP_ALIVE)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--include-unapproved",
        action="store_true",
        help="Include database rows not linked to an approved receipt artifact.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    initialize_database(args.db)
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
        report = ItemEmbeddingIndexer(
            repository=SQLiteSemanticIndexRepository(args.db),
            embedding_client=client,
            batch_size=args.batch_size,
            approved_only=not args.include_unapproved,
        ).rebuild(force=args.force)

    print(json.dumps(report.model_dump(mode="json"), indent=2, ensure_ascii=False))
    return 1 if report.failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
