# RAG-SQL LangGraph query engine

RAG-SQL is the single Ask Your Receipts execution path. It combines semantic
product resolution with validated read-only SQL and hybrid evidence-bound
answer formatting.

## Graph

```text
START
  -> analyze_question
     -> retrieve_entity (zero or more entities)
     -> generate_sql
     -> validate_sql
        -> repair_sql -> validate_sql  [bounded]
     -> execute_sql
     -> extract_answer
        -> finalize_response                    [clear deterministic result]
        -> format_answer_with_llm               [ambiguous reviewed evidence]
           -> validate_formatted_answer
           -> finalize_response
  -> END
```

Terminal branches handle clarification, unsupported questions, missing product
matches, validation exhaustion, execution errors, absent evidence, and rejected
formatter output.

Current identifiers:

```text
engine_version = rag_sql_engine_v2
graph_version  = rag_sql_graph_v2
orchestrator   = langgraph
```

## Safety boundaries

The SQL planner receives a static catalog containing only:

- `analytics_receipts`
- `analytics_purchase_items`

The validator enforces:

- one read-only `SELECT` or `WITH ... SELECT` statement;
- approved views and functions only;
- named parameters for user-derived values;
- protected item-ID parameters produced by semantic resolution;
- deterministic limits for row-shaped results;
- no storage-table access, writes, pragmas, attachments, or extensions.

If validation fails, the planner may receive the exact validation error and
produce a replacement plan. The replacement is validated from the beginning.
The repair budget is controlled by:

```text
RAG_SQL_VALIDATION_REPAIR_COUNT=1
```

Graph recursion is independently bounded:

```text
RAG_SQL_GRAPH_RECURSION_LIMIT=50
```

## Product resolution

Questions about product concepts are analyzed into semantic entities. Hybrid
retrieval returns candidate purchase items, and the candidate resolver selects
concrete `receipt_items.id` values. SQL planning receives those IDs as protected
parameters. It must not replace them with text matching.

## Hybrid answer formatting

The deterministic extractor is always attempted first. Clear monetary,
receipt, product-description, product-type, and explicit-brand answers are
returned without another LLM call.

Only a descriptive result classified as **ambiguous with reviewed evidence**
routes to the bounded answer formatter. The formatter receives only approved
SQL result fields and returns structured JSON:

```json
{
  "schema_version": "rag_sql_answer_format_v1",
  "status": "resolved",
  "values": ["Starbucks"],
  "supporting_item_ids": [175],
  "evidence_fields": ["description", "category_reason"],
  "reason": "The reviewed row explicitly identifies the product brand."
}
```

A deterministic validator then checks:

- every supporting item ID exists in the SQL result;
- every evidence field is approved for the requested operation;
- every value occurs in the cited reviewed fields;
- brand values occur in product-identity evidence, not only merchant context;
- explicit compatible-system, seller, retailer, and merchant roles are rejected;
- no free-form model prose is used as the final answer.

The application renders validated values deterministically. Malformed output,
timeouts, unsupported values, nonexistent item IDs, or unresolved ambiguity end
as `insufficient_info` rather than an invented answer.

Configuration:

```text
RAG_SQL_ANSWER_FORMATTER_ENABLED=1
RAG_SQL_ANSWER_FORMATTER_MODEL=gemma4
RAG_SQL_ANSWER_FORMATTER_NUM_PREDICT=768
RAG_SQL_ANSWER_FORMATTER_TIMEOUT_SECONDS=120
RAG_SQL_ANSWER_FORMATTER_RETRY_COUNT=1
```

The answer formatter shares `RAG_SQL_LLM_NUM_CTX` and
`RAG_SQL_LLM_KEEP_ALIVE` with analysis, candidate resolution, and SQL planning
so Ollama can reuse one resident runner.

## Diagnostics

Responses include:

- graph transition trace;
- stage durations and statuses;
- question analysis;
- retrieval and resolution summaries;
- initial and repaired SQL plans;
- validated SQL metadata;
- SQL execution rows;
- deterministic answer classification;
- whether the fallback formatter was used;
- formatter model status and deterministic validation result;
- aggregated Ollama timing metrics.

Prompts, raw model responses, and embedding vectors are not written to
telemetry.

## Run

```powershell
docker compose `
  -f docker-compose.yml `
  -f docker-compose.dev.yml `
  exec receipt-app `
  python /app/scripts/demo_rag_sql_query.py `
  "Wie viel habe ich für Schuhe ausgegeben?"
```

## Terminal states

- `completed`
- `needs_clarification`
- `not_found`
- `insufficient_info`
- `unsupported`
- `error`
