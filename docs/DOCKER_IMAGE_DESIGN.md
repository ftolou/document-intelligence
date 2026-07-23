# Docker Image Design

## Goal

Avoid rebuilding the 25 GB CUDA/PaddleOCR-VL image whenever application code changes.

## Image layers

```text
receipt-app-runtime:py311
  Python 3.11 slim
  system packages
  requirements/app.txt

paddle-gemma-receipt-app:latest
  FROM receipt-app-runtime:py311
  application source code

receipt-vlm-runtime:cu126
  NVIDIA CUDA 12.6 runtime
  Python
  PaddlePaddle GPU
  PaddleOCR doc-parser
  PaddleX OCR
  Torch/Torchvision cu126
  Transformers stack

paddle-gemma-receipt-vlm:gpu-python-cu126
  FROM receipt-vlm-runtime:cu126
  vlm_service.py
  vl_engine.py
```

## Local workflow

Use already-built images:

```powershell
docker compose up -d --no-build
```

Use source bind mounts for development:

```powershell
docker compose -f docker-compose.yml -f docker-compose.dev.yml up -d --no-build
```

Restart only the changed service:

```powershell
docker compose -f docker-compose.yml -f docker-compose.dev.yml restart receipt-vlm
```

## Build workflow

Build runtime images rarely:

```powershell
.\scripts\docker\build-app-runtime.ps1
.\scripts\docker\build-vlm-runtime.ps1
```

Build thin images often:

```powershell
.\scripts\docker\build-app.ps1
.\scripts\docker\build-vlm.ps1
```

## Why the default compose has no build section

`docker-compose.yml` intentionally references prebuilt images only. This prevents accidental dependency reinstalls when running:

```powershell
docker compose build receipt-vlm
```

Explicit builds are done through scripts or `docker-compose.build.yml`.
