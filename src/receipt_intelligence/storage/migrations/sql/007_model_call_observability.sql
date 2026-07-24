CREATE TABLE IF NOT EXISTS model_calls (
    call_id TEXT PRIMARY KEY,
    recorded_at TEXT NOT NULL,
    started_at TEXT NOT NULL,
    trace_id TEXT,
    job_id TEXT,
    receipt_id TEXT,
    query_id TEXT,
    operation TEXT NOT NULL,
    provider TEXT NOT NULL,
    model TEXT,
    endpoint TEXT NOT NULL,
    status TEXT NOT NULL,
    attempt INTEGER NOT NULL DEFAULT 1,
    duration_ms REAL NOT NULL,
    input_tokens INTEGER,
    output_tokens INTEGER,
    input_characters INTEGER,
    output_characters INTEGER,
    token_source TEXT NOT NULL DEFAULT 'unavailable',
    model_total_duration_ms REAL,
    model_load_duration_ms REAL,
    prompt_evaluation_duration_ms REAL,
    generation_duration_ms REAL,
    configured_context_window INTEGER,
    stop_reason TEXT,
    error TEXT,
    attributes_json TEXT NOT NULL DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_model_calls_recorded_at
    ON model_calls(recorded_at DESC);
CREATE INDEX IF NOT EXISTS idx_model_calls_provider_model
    ON model_calls(provider, model, recorded_at DESC);
CREATE INDEX IF NOT EXISTS idx_model_calls_operation
    ON model_calls(operation, recorded_at DESC);
CREATE INDEX IF NOT EXISTS idx_model_calls_status
    ON model_calls(status, recorded_at DESC);
CREATE INDEX IF NOT EXISTS idx_model_calls_trace
    ON model_calls(trace_id);
CREATE INDEX IF NOT EXISTS idx_model_calls_job
    ON model_calls(job_id);
CREATE INDEX IF NOT EXISTS idx_model_calls_query
    ON model_calls(query_id);

CREATE TABLE IF NOT EXISTS model_pricing (
    provider TEXT NOT NULL,
    model TEXT NOT NULL,
    currency TEXT NOT NULL DEFAULT 'EUR',
    input_price_per_million REAL NOT NULL DEFAULT 0 CHECK(input_price_per_million >= 0),
    output_price_per_million REAL NOT NULL DEFAULT 0 CHECK(output_price_per_million >= 0),
    updated_at TEXT NOT NULL,
    PRIMARY KEY(provider, model)
);
