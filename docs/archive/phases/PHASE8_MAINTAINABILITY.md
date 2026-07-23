# Phase 8: single query engine cutover

Phase 8 removes the retained v1 one-shot query engine after Phase 7 established query
telemetry and a stable regression corpus.

## Removed

- `receipt_intelligence/query/legacy/`
- `query/query_planner.py`
- `query/query_planner_v2.py`
- `query/query_executor.py`
- `query/receipt_qa.py`
- `query/langgraph_engine.py`
- the obsolete v1 planner prompt
- the old `demo_receipt_qa.py` wrapper
- `RECEIPT_QUERY_ENGINE`
- `QUERY_GRAPH_FALLBACK_TO_LEGACY`

## Replacement

- `query/planner.py` owns both LLM and deterministic v2 planning.
- `query/query_tools.py` executes typed receipt-domain operations directly against
  `ReceiptDatabase`.
- `query/service.py` always invokes LangGraph.
- Invalid LLM output falls back to deterministic planning inside the same graph.

## Upgrade behavior

The patch cleanup script removes obsolete modules after the new files have been extracted.
No Python dependency changed, so no app or VLM runtime image rebuild is required in the
bind-mounted development setup.
