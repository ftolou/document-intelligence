"""Runtime configuration for the V14 receipt app."""

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
APP_PORT = int(os.getenv("APP_PORT", "5000"))
DEBUG = os.getenv("FLASK_DEBUG", "0").lower() in {"1", "true", "yes"}

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma4")
OLLAMA_KEEP_ALIVE = os.getenv(
    "OLLAMA_KEEP_ALIVE", ""
)  # empty => omit keep_alive from Ollama request
NUM_CTX = int(os.getenv("NUM_CTX", "24384"))
NUM_PREDICT = int(os.getenv("NUM_PREDICT", "8192"))
LLM_TIMEOUT_SECONDS = float(os.getenv("LLM_TIMEOUT_SECONDS", "300"))
LLM_JSON_RETRY_COUNT = int(os.getenv("LLM_JSON_RETRY_COUNT", "1"))
OLLAMA_FORMAT_JSON = os.getenv("OLLAMA_FORMAT_JSON", "1").lower() in {"1", "true", "yes"}
MAX_LINES_FOR_LLM = int(os.getenv("MAX_LINES_FOR_LLM", "220"))

# RAG semantic item-embedding index. The index is derived and rebuildable;
# approved receipt data in SQLite remains the source of truth.
RAG_EMBEDDING_ENABLED = os.getenv("RAG_EMBEDDING_ENABLED", "1").lower() in {
    "1",
    "true",
    "yes",
    "on",
}
RAG_EMBEDDING_MODEL = os.getenv("RAG_EMBEDDING_MODEL", "embeddinggemma")
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
OCR_MAX_SIDE_LIMIT = int(os.getenv("OCR_MAX_SIDE_LIMIT", "4000"))
OCR_USE_ANGLE_CLS = os.getenv("OCR_USE_ANGLE_CLS", "1").lower() in {"1", "true", "yes"}
OCR_DET_LIMIT_SIDE_LEN = int(os.getenv("OCR_DET_LIMIT_SIDE_LEN", "4000"))
# Safety defaults for PaddleOCR on Windows/CPU. Keep MKLDNN/oneDNN disabled unless you explicitly test it.
OCR_ENABLE_MKLDNN = os.getenv("OCR_ENABLE_MKLDNN", "0").lower() in {"1", "true", "yes"}
OCR_CPU_THREADS = int(os.getenv("OCR_CPU_THREADS", "4"))
OCR_DISABLE_PADDLE_STANDALONE_EXECUTOR = os.getenv(
    "OCR_DISABLE_PADDLE_STANDALONE_EXECUTOR", "0"
).lower() in {"1", "true", "yes"}
OCR_DISABLE_PADDLE_PIR = os.getenv("OCR_DISABLE_PADDLE_PIR", "0").lower() in {"1", "true", "yes"}

VALIDATION_TOLERANCE = float(os.getenv("VALIDATION_TOLERANCE", "0.03"))

# V14.5 optional visual evidence layer. Disabled by default so the normal
# V13-known-good PaddleOCR runtime remains the fast path.
VLM_ENABLED = os.getenv("VLM_ENABLED", "0").lower() in {"1", "true", "yes", "on"}
VLM_BACKEND = os.getenv("VLM_BACKEND", "http_service")
VLM_ALLOW_LOCAL_BACKEND = os.getenv("VLM_ALLOW_LOCAL_BACKEND", "0").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
# For V14.7 two-container mode, the main app calls a separate receipt-vlm HTTP service.
VLM_SERVICE_URL = os.getenv("VLM_SERVICE_URL", "http://receipt-vlm:7870")
# Optional command template, e.g.
# python my_paddleocr_vl_wrapper.py --image "{image}" --out "{output_json}"
VLM_COMMAND = os.getenv("VLM_COMMAND", "")
VLM_TIMEOUT_SECONDS = float(os.getenv("VLM_TIMEOUT_SECONDS", "900"))
VLM_MAX_CHARS = int(os.getenv("VLM_MAX_CHARS", "12000"))

# V14.7.2 VLM service controls. The service resizes very large receipt images before PaddleOCR-VL
# to avoid long CPU stalls/timeouts on full-resolution photos.
VLM_SERVICE_MAX_SIDE_LIMIT = int(os.getenv("VLM_SERVICE_MAX_SIDE_LIMIT", "1600"))
VLM_SERVICE_REQUEST_TIMEOUT_SECONDS = float(
    os.getenv("VLM_SERVICE_REQUEST_TIMEOUT_SECONDS", os.getenv("VLM_TIMEOUT_SECONDS", "900"))
)
VLM_CORRECTION_ENABLED = os.getenv("VLM_CORRECTION_ENABLED", "1").lower() in {
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


# V14.7.3 optional sequential GPU orchestration for Ollama + VLM.
VLM_GPU_ORCHESTRATION = (
    os.getenv("VLM_GPU_ORCHESTRATION", "none").strip().lower()
)  # ignored by V14.10 VLM-first path
OLLAMA_UNLOAD_BEFORE_VLM = os.getenv("OLLAMA_UNLOAD_BEFORE_VLM", "0").lower() in {
    "1",
    "true",
    "yes",
    "on",
}
OLLAMA_RELOAD_AFTER_VLM = os.getenv("OLLAMA_RELOAD_AFTER_VLM", "0").lower() in {
    "1",
    "true",
    "yes",
    "on",
}
OLLAMA_CONTROL_MODE = os.getenv("OLLAMA_CONTROL_MODE", "api").strip().lower()
OLLAMA_CONTROL_TIMEOUT_SECONDS = float(os.getenv("OLLAMA_CONTROL_TIMEOUT_SECONDS", "120"))
OLLAMA_UNLOAD_COMMAND = os.getenv("OLLAMA_UNLOAD_COMMAND", "")
OLLAMA_START_COMMAND = os.getenv("OLLAMA_START_COMMAND", "")
OLLAMA_RELOAD_PROMPT = os.getenv("OLLAMA_RELOAD_PROMPT", "ok")
OLLAMA_GPU_HANDOFF_WAIT_SECONDS = float(os.getenv("OLLAMA_GPU_HANDOFF_WAIT_SECONDS", "3"))

# V14.7.9 PaddleOCR-VL confirmed working local route
VLM_ENGINE = os.getenv("VLM_ENGINE", "transformers").strip()

# V14.14 LLM-first item categorization. Runs after final receipt extraction and validation.
CATEGORIZATION_ENABLED = os.getenv("CATEGORIZATION_ENABLED", "1").lower() in {
    "1",
    "true",
    "yes",
    "on",
}
CATEGORIZATION_MODEL = os.getenv("CATEGORIZATION_MODEL", OLLAMA_MODEL)
CATEGORIZATION_NUM_CTX = int(os.getenv("CATEGORIZATION_NUM_CTX", "8192"))
CATEGORIZATION_NUM_PREDICT = int(os.getenv("CATEGORIZATION_NUM_PREDICT", "4096"))
CATEGORIZATION_TIMEOUT_SECONDS = float(os.getenv("CATEGORIZATION_TIMEOUT_SECONDS", "180"))
CATEGORIZATION_FORMAT_JSON = os.getenv("CATEGORIZATION_FORMAT_JSON", "1").lower() in {
    "1",
    "true",
    "yes",
    "on",
}

# Phase 7 local observability and readiness controls.
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
READINESS_PROBE_VLM = os.getenv("READINESS_PROBE_VLM", "1").lower() in {
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
READINESS_REQUIRE_VLM = os.getenv("READINESS_REQUIRE_VLM", "0").lower() in {
    "1",
    "true",
    "yes",
    "on",
}
READINESS_TIMEOUT_SECONDS = float(os.getenv("READINESS_TIMEOUT_SECONDS", "2"))
