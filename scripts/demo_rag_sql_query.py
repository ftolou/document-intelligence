#!/usr/bin/env python3
"""Run the isolated Step-6 RAG-SQL strategy without changing the web app."""

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
from receipt_intelligence.rag.candidate_resolver import (  # noqa: E402
    CandidateResolverConfig,
)
from receipt_intelligence.rag_sql.answer_formatter import AnswerFormatterConfig  # noqa: E402
from receipt_intelligence.rag_sql.planner import RagSqlPlannerConfig  # noqa: E402
from receipt_intelligence.rag_sql.question_analyzer import (  # noqa: E402
    QuestionAnalyzerConfig,
)
from receipt_intelligence.rag_sql.runtime import (  # noqa: E402
    RagSqlRuntime,
    RagSqlRuntimeConfig,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question", help="Complete receipt analytics question.")
    parser.add_argument("--db", type=Path, default=settings.RECEIPT_DB_PATH)
    parser.add_argument("--ollama-url", default=settings.OLLAMA_URL)
    parser.add_argument("--embedding-model", default=settings.RAG_EMBEDDING_MODEL)
    parser.add_argument("--analyzer-model", default=settings.RAG_SQL_ANALYZER_MODEL)
    parser.add_argument("--resolver-model", default=settings.RAG_CANDIDATE_MODEL)
    parser.add_argument("--planner-model", default=settings.RAG_SQL_PLANNER_MODEL)
    parser.add_argument(
        "--answer-formatter-model",
        default=settings.RAG_SQL_ANSWER_FORMATTER_MODEL,
    )
    parser.add_argument("--retrieval-limit", type=int, default=settings.RAG_SQL_RETRIEVAL_LIMIT)
    parser.add_argument("--max-rows", type=int, default=settings.RAG_SQL_MAX_ROWS)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    shared_num_ctx = settings.RAG_SQL_LLM_NUM_CTX
    keep_alive = (
        settings.RAG_SQL_LLM_KEEP_ALIVE
        or settings.RAG_SQL_KEEP_ALIVE
        or settings.OLLAMA_KEEP_ALIVE
        or None
    )
    strategy = RagSqlRuntime(
        RagSqlRuntimeConfig(
            database_path=args.db,
            ollama_url=args.ollama_url,
            embedding_model=args.embedding_model,
            embedding_timeout_seconds=settings.RAG_EMBEDDING_TIMEOUT_SECONDS,
            embedding_keep_alive=settings.RAG_EMBEDDING_KEEP_ALIVE or None,
            analyzer=QuestionAnalyzerConfig(
                enabled=settings.RAG_SQL_ENABLED,
                ollama_url=args.ollama_url,
                model=args.analyzer_model,
                num_ctx=shared_num_ctx,
                num_predict=settings.RAG_SQL_ANALYZER_NUM_PREDICT,
                timeout_seconds=settings.RAG_SQL_ANALYZER_TIMEOUT_SECONDS,
                retry_count=settings.RAG_SQL_ANALYZER_RETRY_COUNT,
                format_json=settings.RAG_SQL_FORMAT_JSON,
                keep_alive=keep_alive,
                maximum_entities=settings.RAG_SQL_MAX_ENTITIES,
            ),
            resolver=CandidateResolverConfig(
                enabled=settings.RAG_CANDIDATE_RESOLVER_ENABLED,
                ollama_url=args.ollama_url,
                model=args.resolver_model,
                num_ctx=shared_num_ctx,
                num_predict=settings.RAG_CANDIDATE_NUM_PREDICT,
                timeout_seconds=settings.RAG_CANDIDATE_TIMEOUT_SECONDS,
                retry_count=settings.RAG_CANDIDATE_RETRY_COUNT,
                format_json=settings.RAG_CANDIDATE_FORMAT_JSON,
                keep_alive=keep_alive,
                maximum_candidates=settings.RAG_CANDIDATE_MAX_CANDIDATES,
            ),
            planner=RagSqlPlannerConfig(
                enabled=settings.RAG_SQL_ENABLED,
                ollama_url=args.ollama_url,
                model=args.planner_model,
                num_ctx=shared_num_ctx,
                num_predict=settings.RAG_SQL_PLANNER_NUM_PREDICT,
                timeout_seconds=settings.RAG_SQL_PLANNER_TIMEOUT_SECONDS,
                retry_count=settings.RAG_SQL_PLANNER_RETRY_COUNT,
                format_json=settings.RAG_SQL_FORMAT_JSON,
                keep_alive=keep_alive,
                maximum_rows=args.max_rows,
            ),
            answer_formatter=AnswerFormatterConfig(
                enabled=settings.RAG_SQL_ANSWER_FORMATTER_ENABLED,
                ollama_url=args.ollama_url,
                model=args.answer_formatter_model,
                num_ctx=shared_num_ctx,
                num_predict=settings.RAG_SQL_ANSWER_FORMATTER_NUM_PREDICT,
                timeout_seconds=settings.RAG_SQL_ANSWER_FORMATTER_TIMEOUT_SECONDS,
                retry_count=settings.RAG_SQL_ANSWER_FORMATTER_RETRY_COUNT,
                format_json=settings.RAG_SQL_FORMAT_JSON,
                keep_alive=keep_alive,
                maximum_rows=args.max_rows,
            ),
            retrieval_limit=args.retrieval_limit,
            retrieval_maximum_limit=settings.RAG_RETRIEVAL_MAX_LIMIT,
            retrieval_minimum_score=settings.RAG_RETRIEVAL_MINIMUM_SCORE,
            retrieval_rrf_k=settings.RAG_RETRIEVAL_RRF_K,
            retrieval_vector_weight=settings.RAG_RETRIEVAL_VECTOR_WEIGHT,
            retrieval_lexical_weight=settings.RAG_RETRIEVAL_LEXICAL_WEIGHT,
            maximum_rows=args.max_rows,
            execution_timeout_seconds=settings.RAG_SQL_EXECUTION_TIMEOUT_SECONDS,
        )
    )
    response = strategy.execute(args.question)
    print(json.dumps(response.model_dump(mode="json"), ensure_ascii=False, indent=2))
    if response.status == "completed":
        return 0
    if response.status in {"needs_clarification", "not_found", "unsupported"}:
        return 3
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
