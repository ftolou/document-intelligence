"""Retrieve and resolve semantic receipt-item candidates with the local LLM."""

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
from receipt_intelligence.rag.candidate_resolver import (  # noqa: E402
    CandidateResolutionError,
    CandidateResolver,
    CandidateResolverConfig,
)
from receipt_intelligence.rag.embedding_client import OllamaEmbeddingClient  # noqa: E402
from receipt_intelligence.rag.item_retriever import ItemSemanticRetriever  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("entity", help="Semantic product concept, for example Schuhe.")
    parser.add_argument("--question", help="Optional original analytical user question.")
    parser.add_argument("--db", type=Path, default=settings.RECEIPT_DB_PATH)
    parser.add_argument("--ollama-url", default=settings.OLLAMA_URL)
    parser.add_argument("--embedding-model", default=settings.RAG_EMBEDDING_MODEL)
    parser.add_argument("--resolver-model", default=settings.RAG_CANDIDATE_MODEL)
    parser.add_argument("--limit", type=int, default=settings.RAG_CANDIDATE_MAX_CANDIDATES)
    parser.add_argument("--minimum-score", type=float, default=settings.RAG_RETRIEVAL_MINIMUM_SCORE)
    parser.add_argument("--merchant")
    parser.add_argument("--category")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    with OllamaEmbeddingClient(
        base_url=args.ollama_url,
        model=args.embedding_model,
        timeout_seconds=settings.RAG_EMBEDDING_TIMEOUT_SECONDS,
        keep_alive=settings.RAG_EMBEDDING_KEEP_ALIVE or None,
    ) as embedding_client:
        search_result = ItemSemanticRetriever(
            repository=SQLiteSemanticSearchRepository(args.db),
            embedding_client=embedding_client,
            maximum_limit=settings.RAG_RETRIEVAL_MAX_LIMIT,
            deduplicate=True,
            rrf_k=settings.RAG_RETRIEVAL_RRF_K,
            vector_weight=settings.RAG_RETRIEVAL_VECTOR_WEIGHT,
            lexical_weight=settings.RAG_RETRIEVAL_LEXICAL_WEIGHT,
        ).search(
            args.entity,
            limit=args.limit,
            minimum_score=args.minimum_score,
            merchant=args.merchant,
            category=args.category,
        )

    resolver = CandidateResolver(
        CandidateResolverConfig(
            enabled=settings.RAG_CANDIDATE_RESOLVER_ENABLED,
            ollama_url=args.ollama_url,
            model=args.resolver_model,
            num_ctx=settings.RAG_CANDIDATE_NUM_CTX,
            num_predict=settings.RAG_CANDIDATE_NUM_PREDICT,
            timeout_seconds=settings.RAG_CANDIDATE_TIMEOUT_SECONDS,
            retry_count=settings.RAG_CANDIDATE_RETRY_COUNT,
            format_json=settings.RAG_CANDIDATE_FORMAT_JSON,
            keep_alive=settings.RAG_CANDIDATE_KEEP_ALIVE or None,
            maximum_candidates=settings.RAG_CANDIDATE_MAX_CANDIDATES,
        )
    )

    try:
        bundle = resolver.resolve_search_result(
            search_result,
            semantic_entity=args.entity,
            user_question=args.question,
        )
    except CandidateResolutionError as exc:
        print(
            json.dumps(
                {
                    "status": "failed",
                    "error": str(exc),
                    "query": args.entity,
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        return 2

    print(json.dumps(bundle.model_dump(mode="json"), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
