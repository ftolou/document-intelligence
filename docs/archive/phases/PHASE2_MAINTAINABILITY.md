# Maintainability Phase 2: Web and Query Refactor

This phase changes module boundaries without changing the public HTTP API or
receipt-query result schema.

## Flask application factory

`src/receipt_intelligence/app.py` is now only a compatibility entry point. The
real Flask construction lives in:

```text
receipt_intelligence/web/
├── app_factory.py
├── dependencies.py
├── request_parsing.py
└── routes/
    ├── core.py
    ├── jobs.py
    ├── query.py
    ├── receipts.py
    └── review.py
```

`create_app()` accepts an optional `JobStore` and `ReceiptDatabase`, which makes
HTTP integration tests independent of the real `outputs/` and `data/` folders.
Blueprints validate HTTP input and delegate processing to application services.

## Extracted application services

Background execution and review logic no longer live in the Flask module:

```text
receipt_intelligence/services/
├── artifact_service.py
├── job_processing.py
└── review_service.py
```

`JobProcessingService` owns single-receipt and batch execution. `ReviewService`
owns review artifacts, approval changes, review-queue registration, and approved
receipt import.

## Split LangGraph implementation

The former large `langgraph_engine.py` is now a compatibility facade. The active
implementation is divided by responsibility:

```text
receipt_intelligence/query/
├── graph.py                 # graph construction and invocation
├── graph_nodes.py           # planner/validator/executor/controller nodes
├── graph_state.py           # state and runtime limits
├── graph_support.py         # trace and result-selection helpers
├── finalizer.py             # stable API response construction
└── langgraph_engine.py      # compatibility facade
```

The graph still exposes the same `QueryGraphConfig`, `build_receipt_query_graph`,
and `run_langgraph_receipt_query` symbols.

## Legacy query engine

The one-shot planner and executor are now explicitly isolated under:

```text
receipt_intelligence/query/legacy/
├── planner.py
└── executor.py
```

The old public paths remain true module aliases, so existing imports and test
patch targets continue to work:

```python
from receipt_intelligence.query.query_planner import QueryPlannerConfig
from receipt_intelligence.query.receipt_qa import answer_receipt_question
```

## Validation

Phase 2 adds application-factory, API-route, compatibility-facade, and review
service regression tests. Run:

```powershell
python scripts/run_tests.py
python scripts/run_quality_checks.py
```

No runtime dependency changed. With `docker-compose.dev.yml`, only the app
service needs a restart after applying the patch:

```powershell
.\scripts\docker\restart-app.ps1
```
