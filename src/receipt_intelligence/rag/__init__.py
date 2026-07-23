"""Semantic retrieval foundation for receipt intelligence."""

from receipt_intelligence.rag.candidate_models import (
    CANDIDATE_RESOLUTION_SCHEMA_VERSION,
    CandidateDecision,
    CandidateRecord,
    CandidateResolutionBundle,
    CandidateResolutionPayload,
    CandidateResolutionResult,
    EvidenceStrength,
)
from receipt_intelligence.rag.candidate_prompt import build_candidate_resolution_prompt
from receipt_intelligence.rag.candidate_resolver import (
    CandidateResolutionError,
    CandidateResolver,
    CandidateResolverConfig,
    build_candidate_records,
)
from receipt_intelligence.rag.embedding_client import (
    EmbeddingClientError,
    OllamaEmbeddingClient,
)
from receipt_intelligence.rag.hybrid_scoring import (
    build_fts_query,
    lexical_relevance,
    product_identity_key,
    reciprocal_rank_fusion,
)
from receipt_intelligence.rag.item_documents import (
    build_item_embedding_document,
    build_item_embedding_documents,
    clean_semantic_text,
    is_indexable_description,
)
from receipt_intelligence.rag.item_indexer import ItemEmbeddingIndexer
from receipt_intelligence.rag.item_retriever import ItemSemanticRetriever
from receipt_intelligence.rag.models import (
    EmbeddingBatchResult,
    ItemEmbeddingDocument,
    ItemEmbeddingIndexReport,
    RetrievalEvaluationCase,
    RetrievalEvaluationCaseResult,
    RetrievalEvaluationReport,
    SemanticItemMatch,
    SemanticItemSearchResult,
)
from receipt_intelligence.rag.retrieval_evaluator import (
    ItemRetrievalEvaluator,
    load_evaluation_cases,
)
from receipt_intelligence.rag.vector_codec import blob_to_vector, vector_to_blob

__all__ = [
    "CANDIDATE_RESOLUTION_SCHEMA_VERSION",
    "CandidateDecision",
    "CandidateRecord",
    "CandidateResolutionBundle",
    "CandidateResolutionError",
    "CandidateResolutionPayload",
    "CandidateResolutionResult",
    "CandidateResolver",
    "CandidateResolverConfig",
    "build_candidate_records",
    "build_candidate_resolution_prompt",
    "EmbeddingBatchResult",
    "EvidenceStrength",
    "EmbeddingClientError",
    "ItemEmbeddingDocument",
    "ItemEmbeddingIndexReport",
    "ItemEmbeddingIndexer",
    "build_fts_query",
    "lexical_relevance",
    "product_identity_key",
    "reciprocal_rank_fusion",
    "ItemSemanticRetriever",
    "ItemRetrievalEvaluator",
    "OllamaEmbeddingClient",
    "RetrievalEvaluationCase",
    "RetrievalEvaluationCaseResult",
    "RetrievalEvaluationReport",
    "SemanticItemMatch",
    "SemanticItemSearchResult",
    "build_item_embedding_document",
    "build_item_embedding_documents",
    "clean_semantic_text",
    "is_indexable_description",
    "load_evaluation_cases",
    "blob_to_vector",
    "vector_to_blob",
]
