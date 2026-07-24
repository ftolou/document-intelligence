# RAG-SQL runtime lifecycle

The application composes one RAG-SQL object graph when the Flask application
starts. Query execution reuses that graph instead of rebuilding model clients,
retrievers, validators, SQL repositories, and the LangGraph workflow for every
request.

```text
application startup
  -> build settings-backed runtime configuration
  -> create Ollama generation gateway
  -> create process-scoped embedding client
  -> create semantic retriever and SQL executor
  -> compile one orchestration graph
  -> expose ReceiptQueryService

query request
  -> validate HTTP payload
  -> ReceiptQueryService.execute(...)
  -> RagSqlRuntime.execute(...)
  -> existing compiled engine.execute(...)

application shutdown
  -> dispatcher shutdown
  -> ReceiptQueryService.close()
  -> RagSqlRuntime.close()
  -> embedding HTTP session closed
```

## Ownership

`RagSqlRuntime` owns only resources it creates. Injected engines and embedding
clients remain owned by the caller. Construction is fail-safe: if engine
composition fails after a client has been created, the owned client is closed
before the exception escapes.

The runtime is idempotently closeable and supports a context manager for CLI
usage. Calling `execute()` after shutdown raises an explicit error.

## Optional LangGraph boundary

Core RAG-SQL modules do not import LangGraph. Provider-neutral orchestration
contracts live under:

```text
receipt_intelligence.rag_sql.orchestration.contracts
```

The only production module allowed to import `langgraph` is:

```text
receipt_intelligence.rag_sql.orchestration.langgraph
```

`RagSqlEngine` loads the default LangGraph factory only when an engine is
actually composed. Models, planners, validators, storage adapters, runtime
configuration, and compatibility imports remain usable without LangGraph.

## Composition rules

- `RagSqlRuntime.execute()` must only delegate to the prebuilt engine.
- Database migrations happen during application startup, not during queries.
- Package `__init__.py` exports remain lazy.
- Graph-library imports stay confined to the orchestration adapter.
- Long-lived resources are closed by application shutdown or a CLI `finally`
  block/context manager.

These rules are enforced by `scripts/check_rag_sql_composition.py` and unit
tests.
