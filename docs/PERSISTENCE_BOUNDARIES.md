# Persistence boundaries

Semantic indexing, semantic retrieval, and analytical SQL depend on explicit
repository contracts. Feature packages own document construction, ranking,
validation, and orchestration; SQLite adapters own connections, SQL statements,
FTS access, vector serialization storage, and authorizer policy.

## Contracts

- `SemanticIndexRepository` supplies indexable item sources and persists derived vectors.
- `SemanticSearchRepository` supplies structured vector candidates and lexical ranks.
- `AnalyticalQueryRepository` executes already validated, read-only analytical plans.

Concrete SQLite implementations are selected only at entry points and runtime
composition boundaries under `adapters/storage/sqlite`.

## Database initialization

Migrations are explicit startup work through `storage.bootstrap.initialize_database`
or `ReceiptDatabase.initialize`. Indexing, retrieval, and individual query
execution do not run migrations.

## Dependency rule

`rag` and the application-level parts of `rag_sql` must not import `sqlite3`,
connection factories, migration runners, or storage repositories. CI enforces
this through `scripts/check_persistence_boundaries.py`.
