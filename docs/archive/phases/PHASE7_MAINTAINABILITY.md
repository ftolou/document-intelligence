# Phase 7: stabilization, observability, and regression hardening

Phase 7 establishes evidence for deciding when the remaining query fallback can
be removed. It does not change the receipt extraction semantics or expose raw
SQL to the language model.

## Added

- Query IDs and end-to-end duration for every receipt query.
- Planner-call and safe-tool-call timing in the API response.
- Append-only query telemetry in `var/logs/query_events.jsonl`.
- Extraction metrics artifacts alongside the existing stage trace:
  - `<run_id>_extraction_metrics.json`
  - `latest_extraction_metrics.json`
- `/api/readiness` with database, schema, writable runtime storage, Ollama, and
  optional VLM checks.
- A stable natural-language query regression corpus under
  `tests/fixtures/query_cases/`.
- Explicit test profiles through `scripts/run_test_profile.py`.
- Requests dependency constraints and a compatibility verification script.
- Test isolation for runtime-path tests inside Docker Compose.

## Query response metrics

The existing API response now includes fields such as:

```json
{
  "execution": {
    "query_id": "q_...",
    "duration_ms": 38.4,
    "planner_duration_ms": 12.1,
    "replan_count": 0,
    "tool_calls": [
      {
        "step_id": "query_result",
        "tool": "aggregate_receipts",
        "status": "done",
        "duration_ms": 1.8
      }
    ]
  }
}
```

Financial values are still calculated by validated SQLite/Python tools.

## Test profiles

```powershell
python scripts/run_test_profile.py unit
python scripts/run_test_profile.py integration
python scripts/run_test_profile.py regression
python scripts/run_test_profile.py fast
```

The regression profile is deterministic and does not call Ollama.

## Readiness semantics

`/health` is a liveness check. `/api/readiness` verifies whether required local
resources are usable. Database and runtime storage are always required.
External services are optional by default and can be made required with:

```env
READINESS_REQUIRE_OLLAMA=1
READINESS_REQUIRE_VLM=1
```

## Rebuild impact

Phase 7 constrains `urllib3`, `charset-normalizer`, and `chardet` to versions
supported by Requests. Rebuild only the app runtime and thin app image. The
heavy VLM runtime image is unchanged.
