# RAG embedding storage foundation

Version `v1.25.1-rag-step3-embedding-storage` adds the SQLite storage needed for
semantic item embeddings. It does not yet generate vectors, search them, or
change the Ask Your Receipts query path.

## Tables

### `rag_item_embeddings`

One rebuildable embedding record per receipt item and embedding model:

- `item_id`: foreign key to `receipt_items.id`
- `embedding_model`: model that produced the vector
- `embedding_dimension`: vector length
- `document_text`: canonical semantic text that was embedded
- `content_hash`: SHA-256 hash used to detect unchanged content
- `embedding`: float-vector bytes stored as a SQLite BLOB
- `updated_at`: index update timestamp

The primary key is `(item_id, embedding_model)`. Deleting a receipt item removes
its embedding rows through `ON DELETE CASCADE`.

### `rag_index_state`

Stores resumable index-build status:

- index name and embedding model
- detected vector dimension
- indexed and failed row counts
- last processed item ID
- completion timestamp and last error

The embedding index is derived data. Approved receipts and `receipt_items`
remain the source of truth and must not be rolled back when embedding generation
fails.

## Migration

Database schema version 4 creates the two tables and their indexes. The normal
application startup applies the migration automatically. It can also be applied
explicitly:

```powershell
python scripts/migrate_receipt_db.py
```

The next implementation step is the incremental indexer that reads approved
purchase-item rows, builds canonical documents, calls Ollama `/api/embed` in
batches, and upserts vectors into these tables.
