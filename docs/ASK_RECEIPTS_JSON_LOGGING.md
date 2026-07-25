# Ask Your Receipts JSON diagnostics

The **Ask your receipts** form contains an opt-in **Save diagnostic JSON log** checkbox.
It is disabled by default because the generated file can contain sensitive receipt data.

When enabled for a request, the application writes one standalone JSON file to:

```text
var/logs/ask_receipts/
```

In Docker Compose this directory is persisted through the existing `./var:/app/var`
volume mapping, so the same files are available on the host under:

```text
./var/logs/ask_receipts/
```

## Recorded content

Each file uses schema `ask_receipts_diagnostic_v1` and contains:

- request ID, query ID, timestamps, duration, and terminal status;
- the submitted question and effective row limit;
- the complete RAG-SQL response and diagnostics when execution succeeds;
- exception type, message, application error code, and Python traceback when execution fails;
- request-scoped diagnostic events;
- every observed LLM request, including the prompt and structured-output schema;
- every observed raw LLM response and provider-reported timing/token metrics;
- candidate-resolution validation failures for each retry attempt.

This specifically preserves malformed candidate-resolution output before Pydantic rejects it.
For example, a failed run can show whether `strong_contextual` was written into the
`decision` field or whether trailing fields were emitted inside the `decisions` array.

## Operational behavior

Logging is observational:

- an inability to create the JSON file does not fail the receipt query;
- the API response reports `diagnostic_log.saved=false` when writing fails;
- successful and failed queries both return the generated filename when writing succeeds;
- each request receives its own file, avoiding shared JSON-array update races;
- files are written atomically through a temporary file and rename.

## Data handling

The log can include product descriptions, merchant data, the user's question, SQL,
full prompts, and raw model output. Do not enable it for normal operation unless this
information may be stored locally. Delete diagnostic files after troubleshooting.
