CREATE TABLE IF NOT EXISTS rag_item_embeddings (
    item_id INTEGER NOT NULL,
    embedding_model TEXT NOT NULL,
    embedding_dimension INTEGER NOT NULL CHECK (embedding_dimension > 0),
    document_text TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    embedding BLOB NOT NULL,
    updated_at TEXT NOT NULL,

    PRIMARY KEY (item_id, embedding_model),

    FOREIGN KEY (item_id)
        REFERENCES receipt_items(id)
        ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_rag_item_embeddings_model
    ON rag_item_embeddings(embedding_model);

CREATE INDEX IF NOT EXISTS idx_rag_item_embeddings_content_hash
    ON rag_item_embeddings(content_hash);

CREATE TABLE IF NOT EXISTS rag_index_state (
    index_name TEXT PRIMARY KEY,
    embedding_model TEXT NOT NULL,
    embedding_dimension INTEGER CHECK (
        embedding_dimension IS NULL OR embedding_dimension > 0
    ),
    indexed_count INTEGER NOT NULL DEFAULT 0 CHECK (indexed_count >= 0),
    failed_count INTEGER NOT NULL DEFAULT 0 CHECK (failed_count >= 0),
    last_indexed_item_id INTEGER,
    last_completed_at TEXT,
    last_error TEXT
);
