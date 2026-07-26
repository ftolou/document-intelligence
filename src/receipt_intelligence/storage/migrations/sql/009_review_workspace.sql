CREATE TABLE IF NOT EXISTS receipt_review_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id TEXT NOT NULL,
    receipt_db_id INTEGER REFERENCES receipts(id) ON DELETE SET NULL,
    revision INTEGER NOT NULL,
    requested_status TEXT,
    effective_status TEXT NOT NULL,
    reviewer TEXT,
    notes TEXT,
    changed_fields_json TEXT NOT NULL,
    validation_json TEXT NOT NULL,
    receipt_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE(job_id, revision)
);

CREATE INDEX IF NOT EXISTS idx_receipt_review_history_job_revision
    ON receipt_review_history(job_id, revision DESC);
CREATE INDEX IF NOT EXISTS idx_receipt_review_history_receipt
    ON receipt_review_history(receipt_db_id, revision DESC);
CREATE INDEX IF NOT EXISTS idx_review_queue_status_updated
    ON review_queue(queue_status, updated_at DESC);
