# Observability and readiness

## Query telemetry

Query telemetry is enabled by default and written to:

```text
var/logs/query_events.jsonl
```

Each line is an independent `query_execution_event_v2` JSON object. It contains
query engine, planner source, plan size, tool timings, replan status, and
errors. Receipt data rows are not copied into the telemetry event.

Disable persistence while retaining metrics in API responses:

```env
QUERY_TELEMETRY_ENABLED=0
```

## Extraction metrics

Each receipt job writes:

```text
<run_id>_extraction_stage_trace.json
<run_id>_extraction_metrics.json
```

The metrics file contains UTC timestamps, stage durations, completion counts,
and the final extraction status.

## Endpoints

```text
GET /health
GET /api/readiness
```

`/health` only confirms that Flask is alive. `/api/readiness` returns HTTP 503
when a required check fails.

## Dependency verification

After rebuilding the app runtime, run:

```powershell
docker compose -f docker-compose.yml -f docker-compose.dev.yml exec receipt-app `
  python /app/scripts/check_dependency_compatibility.py
```
