# Model-call usage and cost dashboard

The Models tab records one row for every instrumented text-generation request.
It is intended to answer three questions:

1. How many input and output tokens did each call use?
2. How long did prompt evaluation and generation take?
3. What would the calls cost under a configured provider price?

## Recorded fields

The Ollama adapter reports prompt tokens, generated tokens, total request time,
model load time, prompt-evaluation time, generation time, stop reason, model,
operation and attempt. Calls are correlated with extraction job IDs or query IDs.
Failed calls are recorded as well, without changing exception behavior.

The stable operation names include `receipt_main_parse`,
`receipt_table_interpretation`, `receipt_patch_repair`,
`receipt_item_categorization`, `rag_sql_question_analysis`,
`rag_candidate_resolution`, `rag_sql_planning`, and
`rag_sql_answer_formatting`.

## Pricing

Prices are entered in the Models tab per provider and exact model name as input
and output price per one million tokens. Prices are not hardcoded because cloud
pricing changes and local Ollama calls have no direct API price. The dashboard
uses the current configured price to estimate all matching historical calls.
This makes the result a scenario estimate rather than an accounting invoice.

## Storage and privacy

Model-call rows are stored in the application SQLite database. Prompts and full
outputs are not persisted. Only character counts, token counts, timing,
correlation identifiers, status and errors are retained.

Set `MODEL_CALL_TELEMETRY_ENABLED=0` to disable new model-call rows. Existing
rows remain available to the dashboard.
