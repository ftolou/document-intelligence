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


## Extraction strategy

`extraction_strategy` accepts:

- `current`: the existing compact OCR/VLM/table path.
- `spatial_overview`: the experimental geometry-first path described in `SPATIAL_OVERVIEW_EXTRACTION.md`.

The active spatial tuning field is:

- `spatial_canvas_width`

The historical fields `spatial_overview_num_ctx`, `spatial_overview_num_predict`, and `spatial_overview_timeout_seconds` remain accepted for configuration compatibility, but the spatial strategy no longer performs a separate overview LLM call and therefore does not use them.

The corresponding environment variables use uppercase names, for example `EXTRACTION_STRATEGY=spatial_overview`. Invalid strategies and unsafe canvas settings are rejected by the immutable configuration contract.
