CREATE TABLE IF NOT EXISTS schema_meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS receipts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id TEXT UNIQUE,
    merchant_name TEXT,
    merchant_normalized TEXT,
    receipt_date TEXT,
    receipt_time TEXT,
    currency TEXT,
    subtotal REAL,
    tax_total REAL,
    grand_total REAL,
    paid_total REAL,
    payment_method TEXT,
    review_status TEXT,
    reviewer TEXT,
    image_path TEXT,
    approved_receipt_path TEXT,
    source_receipt_path TEXT,
    raw_json TEXT NOT NULL,
    file_sha256 TEXT,
    content_fingerprint TEXT,
    duplicate_status TEXT,
    duplicate_of_receipt_id INTEGER,
    duplicate_score REAL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS receipt_items (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    receipt_id INTEGER NOT NULL REFERENCES receipts(id) ON DELETE CASCADE,
    item_index INTEGER NOT NULL,
    raw_name TEXT,
    normalized_name TEXT,
    category TEXT,
    parser_item_type TEXT,
    category_group TEXT,
    category_key TEXT,
    category_reason TEXT,
    semantic_description TEXT,
    quantity REAL,
    unit TEXT,
    unit_price REAL,
    original_price REAL,
    discount_amount REAL,
    line_total REAL,
    tax_code TEXT,
    vat_rate TEXT,
    confidence REAL,
    review_status TEXT,
    embedding_text TEXT NOT NULL,
    raw_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS product_aliases (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    alias TEXT UNIQUE NOT NULL,
    normalized_name TEXT,
    category TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS review_queue (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id TEXT UNIQUE NOT NULL,
    queue_status TEXT NOT NULL,
    decision TEXT,
    balanced INTEGER,
    difference REAL,
    issue_count INTEGER,
    merchant_name TEXT,
    merchant_normalized TEXT,
    receipt_date TEXT,
    receipt_time TEXT,
    grand_total REAL,
    item_count INTEGER,
    file_sha256 TEXT,
    content_fingerprint TEXT,
    item_signature TEXT,
    image_path TEXT,
    final_receipt_path TEXT,
    review_record_path TEXT,
    receipt_db_id INTEGER,
    duplicate_status TEXT,
    duplicate_score REAL,
    duplicate_candidates_json TEXT,
    raw_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS duplicate_candidates (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id TEXT NOT NULL,
    candidate_job_id TEXT,
    candidate_receipt_db_id INTEGER,
    score REAL NOT NULL,
    status TEXT NOT NULL DEFAULT 'candidate',
    reason_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    resolved_at TEXT,
    UNIQUE(job_id, candidate_job_id, candidate_receipt_db_id)
);
