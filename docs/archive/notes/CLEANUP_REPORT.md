# Cleanup Report

## Kept

- Current app runtime files: `app.py`, `integrated_receipt_pipeline.py`, OCR/LLM/VLM modules, validation, correction, re-OCR, visual evidence, and categorization modules.
- Current V14.14.1.1 version file.
- Flask frontend in `static/`.
- Stable main app Dockerfile.
- Heavy CUDA 12.6 GPU VLM Dockerfile.
- One canonical `docker-compose.yml` for the VLM baseline.
- Runtime folders with `.gitkeep`: `uploads/`, `outputs/`, `batch_input/`, `model_cache/`.

## Removed as obsolete/noisy

- `.venv/`
- `.idea/`
- `__pycache__/`
- Old V14 patch readmes and manifests.
- Old CPU VLM and experimental VLM Docker/compose variants.
- Old alternative VLM requirements files.
- Bundled sample receipt photos in `batch_input/fotos/`.
- Nested/duplicated packaging structure from the uploaded ZIP.

## Compose simplification

Before cleanup there were multiple compose choices:

- `docker-compose.yml` — app only, VLM disabled
- `docker-compose.vlm.yml` — old one-container VLM experiment
- `docker-compose.vlm-service.yml` — CPU/non-GPU VLM service
- `docker-compose.vlm-gpu.yml` — GPU VLM service

After cleanup there is only one default baseline:

```powershell
docker compose up --no-build
```

It starts the app with VLM enabled and uses `paddle-gemma-receipt-vlm:gpu-cli-cu126`.
