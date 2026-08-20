"""Runtime configuration for the receipt intelligence application."""

from __future__ import annotations

import os
from pathlib import Path

from receipt_intelligence.runtime.paths import RuntimePaths

PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(os.getenv("APP_PROJECT_ROOT", Path.cwd())).resolve()
if not (PROJECT_ROOT / "VERSION").exists() and (PACKAGE_DIR.parents[1] / "VERSION").exists():
    PROJECT_ROOT = PACKAGE_DIR.parents[1]
BASE_DIR = PROJECT_ROOT
STATIC_DIR = Path(os.getenv("STATIC_DIR", BASE_DIR / "static"))

RUNTIME_PATHS = RuntimePaths.from_environment(BASE_DIR)
RUNTIME_PATHS.ensure_directories()
RUNTIME_LAYOUT = RUNTIME_PATHS.layout
VAR_DIR = RUNTIME_PATHS.var_root
UPLOAD_DIR = RUNTIME_PATHS.uploads_dir
RESULTS_DIR = RUNTIME_PATHS.jobs_dir
BATCH_INPUT_DIR = RUNTIME_PATHS.batch_input_dir
DATA_DIR = RUNTIME_PATHS.database_dir
REPORTS_DIR = RUNTIME_PATHS.reports_dir
LOGS_DIR = RUNTIME_PATHS.logs_dir
RECEIPT_DB_PATH = RUNTIME_PATHS.receipt_db_path
APP_HOST = os.getenv("APP_HOST", "0.0.0.0")
APP_PORT = int(os.getenv("APP_PORT", "7860"))
DEBUG = os.getenv("FLASK_DEBUG", "0").lower() in {"1", "true", "yes"}

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma4:latest")
QWEN_TRANSCRIPTION_MODEL = os.getenv("QWEN_TRANSCRIPTION_MODEL", "qwen3.5:latest")
OLLAMA_KEEP_ALIVE = os.getenv(
    "OLLAMA_KEEP_ALIVE", ""
)  # empty => omit keep_alive from Ollama request
NUM_CTX = int(os.getenv("NUM_CTX", "16384"))
NUM_PREDICT = int(os.getenv("NUM_PREDICT", "8192"))
LLM_TIMEOUT_SECONDS = float(os.getenv("LLM_TIMEOUT_SECONDS", "300"))
LLM_JSON_RETRY_COUNT = int(os.getenv("LLM_JSON_RETRY_COUNT", "1"))
OLLAMA_FORMAT_JSON = os.getenv("OLLAMA_FORMAT_JSON", "1").lower() in {"1", "true", "yes"}

EXTRACTION_BACKEND = os.getenv("EXTRACTION_BACKEND", "local_specialized").strip().lower()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_RECEIPT_MODEL = os.getenv("OPENAI_RECEIPT_MODEL", "gpt-5.6-luna").strip()
OPENAI_RECEIPT_REASONING_EFFORT = (
    os.getenv("OPENAI_RECEIPT_REASONING_EFFORT", "medium").strip().lower()
)
OPENAI_RECEIPT_IMAGE_DETAIL = os.getenv("OPENAI_RECEIPT_IMAGE_DETAIL", "high").strip().lower()
OPENAI_RECEIPT_MAX_OUTPUT_TOKENS = int(os.getenv("OPENAI_RECEIPT_MAX_OUTPUT_TOKENS", "12000"))
OPENAI_RECEIPT_TIMEOUT_SECONDS = float(os.getenv("OPENAI_RECEIPT_TIMEOUT_SECONDS", "180"))

# RAG semantic item-embedding index. The index is derived and rebuildable;
# approved receipt data in SQLite remains the source of truth.
RAG_EMBEDDING_ENABLED = os.getenv("RAG_EMBEDDING_ENABLED", "1").lower() in {
    "1",
    "true",
    "yes",
    "on",
}
RAG_EMBEDDING_PROVIDER = os.getenv("RAG_EMBEDDING_PROVIDER", "ollama").strip().lower()
_default_rag_embedding_model = (
    "text-embedding-3-small"
    if RAG_EMBEDDING_PROVIDER == "openai"
    else "embeddinggemma:latest"
)
RAG_EMBEDDING_MODEL = os.getenv("RAG_EMBEDDING_MODEL", _default_rag_embedding_model).strip()
_rag_embedding_base_url = os.getenv("RAG_EMBEDDING_BASE_URL", "").strip()
RAG_EMBEDDING_BASE_URL = (
    _rag_embedding_base_url
    or (OLLAMA_URL if RAG_EMBEDDING_PROVIDER == "ollama" else None)
)
_rag_embedding_dimensions = os.getenv("RAG_EMBEDDING_DIMENSIONS", "").strip()
RAG_EMBEDDING_DIMENSIONS = (
    int(_rag_embedding_dimensions) if _rag_embedding_dimensions else None
)
RAG_EMBEDDING_BATCH_SIZE = int(os.getenv("RAG_EMBEDDING_BATCH_SIZE", "32"))
RAG_EMBEDDING_TIMEOUT_SECONDS = float(os.getenv("RAG_EMBEDDING_TIMEOUT_SECONDS", "120"))
RAG_EMBEDDING_KEEP_ALIVE = os.getenv("RAG_EMBEDDING_KEEP_ALIVE", "30m")
RAG_RETRIEVAL_DEFAULT_LIMIT = int(os.getenv("RAG_RETRIEVAL_DEFAULT_LIMIT", "10"))
RAG_RETRIEVAL_MAX_LIMIT = int(os.getenv("RAG_RETRIEVAL_MAX_LIMIT", "100"))
_rag_retrieval_minimum_score = os.getenv("RAG_RETRIEVAL_MINIMUM_SCORE", "").strip()
RAG_RETRIEVAL_MINIMUM_SCORE = (
    float(_rag_retrieval_minimum_score) if _rag_retrieval_minimum_score else None
)
RAG_RETRIEVAL_RRF_K = int(os.getenv("RAG_RETRIEVAL_RRF_K", "60"))
RAG_RETRIEVAL_VECTOR_WEIGHT = float(os.getenv("RAG_RETRIEVAL_VECTOR_WEIGHT", "1.0"))
RAG_RETRIEVAL_LEXICAL_WEIGHT = float(os.getenv("RAG_RETRIEVAL_LEXICAL_WEIGHT", "1.5"))
RAG_RETRIEVAL_DEDUPLICATE = os.getenv("RAG_RETRIEVAL_DEDUPLICATE", "1").lower() in {
    "1",
    "true",
    "yes",
    "on",
}
RAG_CANDIDATE_RESOLVER_ENABLED = os.getenv("RAG_CANDIDATE_RESOLVER_ENABLED", "1").lower() in {
    "1",
    "true",
    "yes",
    "on",
}
RAG_CANDIDATE_MODEL = os.getenv("RAG_CANDIDATE_MODEL", OLLAMA_MODEL)
RAG_CANDIDATE_MAX_CANDIDATES = int(os.getenv("RAG_CANDIDATE_MAX_CANDIDATES", "12"))
# Step 7.1.2 uses one stable Gemma runner configuration across all RAG-SQL
# LLM stages. A shared context size prevents Ollama from recreating the same
# model runner merely because analysis, resolution, and planning requested
# different context windows. The stage-specific value remains available to
# standalone candidate-resolution tools.
RAG_SQL_LLM_NUM_CTX = int(os.getenv("RAG_SQL_LLM_NUM_CTX", "6144"))
RAG_CANDIDATE_NUM_CTX = int(os.getenv("RAG_CANDIDATE_NUM_CTX", str(RAG_SQL_LLM_NUM_CTX)))
RAG_CANDIDATE_NUM_PREDICT = int(os.getenv("RAG_CANDIDATE_NUM_PREDICT", "1536"))
RAG_CANDIDATE_TIMEOUT_SECONDS = float(os.getenv("RAG_CANDIDATE_TIMEOUT_SECONDS", "120"))
RAG_CANDIDATE_RETRY_COUNT = int(os.getenv("RAG_CANDIDATE_RETRY_COUNT", "1"))
RAG_CANDIDATE_FORMAT_JSON = os.getenv("RAG_CANDIDATE_FORMAT_JSON", "1").lower() in {
    "1",
    "true",
    "yes",
    "on",
}
RAG_CANDIDATE_KEEP_ALIVE = os.getenv("RAG_CANDIDATE_KEEP_ALIVE", OLLAMA_KEEP_ALIVE)

# RAG-assisted SQL query engine orchestrated by LangGraph.
RAG_SQL_ENABLED = os.getenv("RAG_SQL_ENABLED", "1").lower() in {
    "1",
    "true",
    "yes",
    "on",
}
RAG_SQL_ANALYZER_MODEL = os.getenv("RAG_SQL_ANALYZER_MODEL", OLLAMA_MODEL)
RAG_SQL_ANALYZER_NUM_CTX = int(os.getenv("RAG_SQL_ANALYZER_NUM_CTX", str(RAG_SQL_LLM_NUM_CTX)))
RAG_SQL_ANALYZER_NUM_PREDICT = int(os.getenv("RAG_SQL_ANALYZER_NUM_PREDICT", "768"))
RAG_SQL_ANALYZER_TIMEOUT_SECONDS = float(os.getenv("RAG_SQL_ANALYZER_TIMEOUT_SECONDS", "120"))
RAG_SQL_ANALYZER_RETRY_COUNT = int(os.getenv("RAG_SQL_ANALYZER_RETRY_COUNT", "1"))
RAG_SQL_PLANNER_MODEL = os.getenv("RAG_SQL_PLANNER_MODEL", OLLAMA_MODEL)
RAG_SQL_PLANNER_NUM_CTX = int(os.getenv("RAG_SQL_PLANNER_NUM_CTX", str(RAG_SQL_LLM_NUM_CTX)))
RAG_SQL_PLANNER_NUM_PREDICT = int(os.getenv("RAG_SQL_PLANNER_NUM_PREDICT", "2048"))
RAG_SQL_PLANNER_TIMEOUT_SECONDS = float(os.getenv("RAG_SQL_PLANNER_TIMEOUT_SECONDS", "120"))
RAG_SQL_PLANNER_RETRY_COUNT = int(os.getenv("RAG_SQL_PLANNER_RETRY_COUNT", "1"))
RAG_SQL_ANSWER_FORMATTER_ENABLED = os.getenv("RAG_SQL_ANSWER_FORMATTER_ENABLED", "1").lower() in {
    "1",
    "true",
    "yes",
    "on",
}
RAG_SQL_ANSWER_FORMATTER_MODEL = os.getenv("RAG_SQL_ANSWER_FORMATTER_MODEL", OLLAMA_MODEL)
RAG_SQL_ANSWER_FORMATTER_NUM_PREDICT = int(os.getenv("RAG_SQL_ANSWER_FORMATTER_NUM_PREDICT", "768"))
RAG_SQL_ANSWER_FORMATTER_TIMEOUT_SECONDS = float(
    os.getenv("RAG_SQL_ANSWER_FORMATTER_TIMEOUT_SECONDS", "120")
)
RAG_SQL_ANSWER_FORMATTER_RETRY_COUNT = int(os.getenv("RAG_SQL_ANSWER_FORMATTER_RETRY_COUNT", "1"))
RAG_SQL_FORMAT_JSON = os.getenv("RAG_SQL_FORMAT_JSON", "1").lower() in {
    "1",
    "true",
    "yes",
    "on",
}
# Keep the RAG-SQL LLM resident across analysis, candidate resolution, and
# SQL planning. ``RAG_SQL_KEEP_ALIVE`` remains a backwards-compatible alias.
RAG_SQL_LLM_KEEP_ALIVE = os.getenv(
    "RAG_SQL_LLM_KEEP_ALIVE",
    os.getenv("RAG_SQL_KEEP_ALIVE", "30m"),
)
RAG_SQL_KEEP_ALIVE = RAG_SQL_LLM_KEEP_ALIVE
RAG_SQL_MAX_ENTITIES = int(os.getenv("RAG_SQL_MAX_ENTITIES", "4"))
RAG_SQL_RETRIEVAL_LIMIT = int(os.getenv("RAG_SQL_RETRIEVAL_LIMIT", "12"))
RAG_SQL_MAX_ROWS = int(os.getenv("RAG_SQL_MAX_ROWS", "100"))
RAG_SQL_EXECUTION_TIMEOUT_SECONDS = float(os.getenv("RAG_SQL_EXECUTION_TIMEOUT_SECONDS", "5"))
RAG_SQL_VALIDATION_REPAIR_COUNT = int(os.getenv("RAG_SQL_VALIDATION_REPAIR_COUNT", "1"))
RAG_SQL_GRAPH_RECURSION_LIMIT = int(os.getenv("RAG_SQL_GRAPH_RECURSION_LIMIT", "50"))

OCR_LANG = os.getenv("OCR_LANG", "german")
OCR_DEVICE = os.getenv("OCR_DEVICE", "cpu")
EXTRACTION_MAX_CROPS = max(1, int(os.getenv("EXTRACTION_MAX_CROPS", "4")))

VALIDATION_TOLERANCE = float(os.getenv("VALIDATION_TOLERANCE", "0.03"))
CORRECTION_ENABLED = os.getenv("CORRECTION_ENABLED", "1").lower() in {
    "1",
    "true",
    "yes",
    "on",
}

MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "25"))
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp", "bmp", "tif", "tiff"}

# Server-side batch folder execution. In Docker Compose this maps to ./var/batch_input:/app/var/batch_input.
BATCH_MAX_FILES = int(os.getenv("BATCH_MAX_FILES", "250"))
BATCH_RECURSIVE_DEFAULT = os.getenv("BATCH_RECURSIVE_DEFAULT", "0").lower() in {
    "1",
    "true",
    "yes",
    "on",
}
BATCH_ALLOW_ANY_PATH = os.getenv("BATCH_ALLOW_ANY_PATH", "0").lower() in {"1", "true", "yes", "on"}

# Bounded local background worker. One worker is the safe default because Paddle
# geometry and Ollama generation share constrained local compute resources.
JOB_WORKER_MAX_WORKERS = max(1, int(os.getenv("JOB_WORKER_MAX_WORKERS", "1")))
JOB_QUEUE_CAPACITY = max(0, int(os.getenv("JOB_QUEUE_CAPACITY", "32")))
JOB_CLAIM_LEASE_SECONDS = max(30.0, float(os.getenv("JOB_CLAIM_LEASE_SECONDS", "120")))
JOB_MAINTENANCE_INTERVAL_SECONDS = max(
    1.0,
    float(os.getenv("JOB_MAINTENANCE_INTERVAL_SECONDS", "10")),
)
JOB_RECOVER_PENDING = os.getenv("JOB_RECOVER_PENDING", "1").lower() in {
    "1",
    "true",
    "yes",
    "on",
}


# LLM-first item categorization runs after final receipt extraction and validation.
CATEGORIZATION_ENABLED = os.getenv("CATEGORIZATION_ENABLED", "1").lower() in {
    "1",
    "true",
    "yes",
    "on",
}
CATEGORIZATION_MODEL = os.getenv("CATEGORIZATION_MODEL", OLLAMA_MODEL)
CATEGORIZATION_NUM_CTX = int(os.getenv("CATEGORIZATION_NUM_CTX", str(NUM_CTX)))
CATEGORIZATION_NUM_PREDICT = int(os.getenv("CATEGORIZATION_NUM_PREDICT", "4096"))
CATEGORIZATION_TIMEOUT_SECONDS = float(os.getenv("CATEGORIZATION_TIMEOUT_SECONDS", "180"))
CATEGORIZATION_FORMAT_JSON = os.getenv("CATEGORIZATION_FORMAT_JSON", "1").lower() in {
    "1",
    "true",
    "yes",
    "on",
}

# Local observability and readiness controls.
QUERY_TELEMETRY_ENABLED = os.getenv("QUERY_TELEMETRY_ENABLED", "1").lower() in {
    "1",
    "true",
    "yes",
    "on",
}
QUERY_TELEMETRY_PATH = Path(
    os.getenv("QUERY_TELEMETRY_PATH", str(LOGS_DIR / "query_events.jsonl"))
).resolve()
READINESS_PROBE_OLLAMA = os.getenv("READINESS_PROBE_OLLAMA", "1").lower() in {
    "1",
    "true",
    "yes",
    "on",
}
READINESS_REQUIRE_OLLAMA = os.getenv("READINESS_REQUIRE_OLLAMA", "0").lower() in {
    "1",
    "true",
    "yes",
    "on",
}
READINESS_TIMEOUT_SECONDS = float(os.getenv("READINESS_TIMEOUT_SECONDS", "2"))

MODEL_CALL_TELEMETRY_ENABLED = os.getenv("MODEL_CALL_TELEMETRY_ENABLED", "1").lower() in {
    "1",
    "true",
    "yes",
    "on",
}
