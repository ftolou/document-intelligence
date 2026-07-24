# Extraction configuration contract

Receipt extraction has two entry points:

- `run_receipt_extraction(ExtractionRequest(...))` is the typed application API.
- `run_integrated_receipt_pipeline(...)` is a compatibility adapter for historical callers.

`ExtractionRequest` is immutable and declares every supported option. Unknown fields are rejected instead of being stored or silently ignored.

The compatibility adapter recognizes only these historical aliases:

| Historical name | Canonical field |
|---|---|
| `vlm_gpu_orchestration` | `gpu_orchestration` |
| `gpu_orchestration_mode` | `gpu_orchestration` |
| `ollama_unload_before_vlm` | `unload_llm_before_vlm` |
| `unload_before_vlm` | `unload_llm_before_vlm` |
| `ollama_reload_after_vlm` | `reload_llm_after_vlm` |
| `reload_after_vlm` | `reload_llm_after_vlm` |

Supplying both an alias and its canonical name is an error. Internal application services must use the canonical typed API rather than the compatibility adapter.
