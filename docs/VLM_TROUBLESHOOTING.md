# VLM Troubleshooting

## Python 3.11 application code appears in the VLM traceback

The VLM service must not import `receipt_intelligence`. Its image and development
mount contain only `services/receipt-vlm/src`.

Check the running command and mounts:

```powershell
docker inspect document-intelligence-pipeline-receipt-vlm-1 `
  --format '{{json .Config.Cmd}} {{json .Mounts}}'
```

Expected entrypoint:

```text
python -m receipt_vlm_service.app
```

Validate the repository boundary:

```powershell
python scripts/check_vlm_architecture.py
```

## Restore the known-working Python 3.10 VLM image

The application remains Python 3.11, but the CUDA/Paddle VLM runtime is Python
3.10. Do not run Ruff `py311` fixes against the standalone service.

For source-only changes, rebuild only the thin image from the existing runtime:

```powershell
.\scripts\docker\build-vlm.ps1
docker compose -f docker-compose.yml -f docker-compose.dev.yml up -d `
  --no-build --force-recreate --no-deps receipt-vlm
```

Verify:

```powershell
docker compose -f docker-compose.yml -f docker-compose.dev.yml exec receipt-vlm `
  python -c "import sys; print(sys.version)"
```

Expected: Python 3.10.x.

## PaddleOCR CLI failure

The HTTP service invokes the established command:

```text
paddleocr doc_parser -i IMAGE --save_path OUT --device gpu:0 --engine transformers
```

Inspect the returned `stderr_tail`, `command`, and `produced_files`. Changes to
Torch, Triton, PaddleOCR, PaddleX, or system packages require comparison with the
known-working `receipt-vlm-runtime:cu126` image before rebuilding it.
