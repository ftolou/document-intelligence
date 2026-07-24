# Observability and readiness

## Event boundary

Extraction and query workflows emit immutable, provider-neutral application
events through `application.ports.events.EventSink`. Feature code does not know
whether an event is written to JSON, JSONL, another telemetry backend, or
ignored. Concrete file adapters live under `adapters/observability/` and are
selected only by application composition.

The event flow is:

```text
extraction / RAG-SQL application service
        ↓ typed application event
application EventSink port
        ↓
JSON snapshot or JSONL adapter
```

Observability adapters must not import extraction contexts, RAG-SQL graph
classes, storage repositories, or Flask transports. Compatibility modules under
`observability/` serialize neutral events but do not own application behavior.

## Query telemetry

Query telemetry is enabled by default and written to:

```text
var/logs/query_events.jsonl
```

Each line is an independent `query_execution_event_v6` JSON object. It contains
query-engine metadata, stage timings, bounded validation/repair counts, result
cardinality, errors, and a provider-neutral `model_calls` summary. Receipt data
rows and model prompts are not copied into telemetry events.

The model summary includes a provider count so Ollama, an OpenAI-compatible
runtime, or another adapter can use the same event schema. Historical
`ollama_summary` diagnostics may still appear in API responses for compatibility,
but new telemetry records use `model_calls`.

Disable persistence while retaining diagnostics in API responses:

```env
QUERY_TELEMETRY_ENABLED=0
```

## Extraction metrics

Each receipt job writes:

```text
<run_id>_extraction_stage_trace.json
<run_id>_extraction_metrics.json
```

The metrics snapshot is an `extraction_metrics_v2` event containing UTC
timestamps, stage durations, completion counts, errors, and the final extraction
status. `latest_extraction_metrics.json` is updated atomically as an alias.

## Readiness

Readiness belongs to runtime operations rather than observability serialization.
The implementation lives under `runtime/readiness.py`; the historical
`observability.readiness` import remains as a compatibility export.

```text
GET /health
GET /api/readiness
```

`/health` only confirms that Flask is alive. `/api/readiness` returns HTTP 503
when a required check fails.

## Boundary verification

Run the observability dependency check directly:

```powershell
python scripts/check_observability_boundaries.py
```

Or run the complete quality suite:

```powershell
python scripts/run_quality_checks.py
```

After rebuilding the app runtime, verify installed dependencies with:

```powershell
docker compose -f docker-compose.yml -f docker-compose.dev.yml exec receipt-app `
  python /app/scripts/check_dependency_compatibility.py
```
