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

`run_receipt_extraction(ExtractionRequest(...))` is the only supported extraction entry point.
Historical OCR/VLM keyword arguments are no longer part of the public API.

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
