# Extraction configuration contract

The canonical API is:

```python
run_receipt_extraction(
    ExtractionRequest(
        source_image_path=image_path,
        result_dir=result_dir,
        run_id=run_id,
        ollama_url=ollama_url,
        model=gemma_model,
    )
)
```

`ExtractionRequest` is immutable and image-first. It has no OCR-JSON input, VLM-service setting,
spatial-parser setting, or extraction-strategy toggle.

`run_integrated_receipt_pipeline(...)` remains temporarily available for historical keyword
callers. It requires `source_image_path`, ignores explicitly declared historical OCR/VLM tuning
arguments, rejects unknown arguments, emits `DeprecationWarning`, and always runs the canonical
workflow.

Important environment settings:

| Setting | Purpose |
|---|---|
| `OLLAMA_URL` | Ollama API base URL |
| `OLLAMA_MODEL` | Gemma extraction/correction model |
| `QWEN_TRANSCRIPTION_MODEL` | Qwen multimodal transcription model |
| `EXTRACTION_MAX_CROPS` | Maximum safe transcription crops |
| `OCR_LANG` / `OCR_DEVICE` | Paddle text-detection configuration |
| `VALIDATION_TOLERANCE` | Receipt validation tolerance |
| `CORRECTION_ENABLED` | Enable specialist correction |
