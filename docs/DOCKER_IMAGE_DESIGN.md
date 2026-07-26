# Docker Image Design

## Goal

Keep the application and GPU VLM service independently buildable. Application
changes must not rebuild or alter the known-working PaddleOCR-VL runtime.

## Image layers

```text
receipt-app-runtime:py311
  Python 3.11
  application dependencies

paddle-gemma-receipt-app:latest
  FROM receipt-app-runtime:py311
  src/receipt_intelligence

receipt-vlm-runtime:cu126
  NVIDIA CUDA 12.6 runtime
  Ubuntu 22.04 Python 3.10
  PaddlePaddle GPU 3.2.1
  PaddleOCR/PaddleX
  Torch/Torchvision cu126

paddle-gemma-receipt-vlm:gpu-python-cu126
  FROM receipt-vlm-runtime:cu126
  services/receipt-vlm/src only
```

The VLM thin image intentionally does not copy `src/receipt_intelligence`.

## Local workflow

Use already-built images:

```powershell
docker compose -f docker-compose.yml -f docker-compose.dev.yml up -d --no-build
```

The development override mounts only `services/receipt-vlm/src` into the VLM
container. Main-application code is not visible to that process.

## Build workflow

Rebuild the heavy VLM runtime only when its CUDA/system/Python dependencies or
`requirements/vlm-gpu-cu126.txt` change:

```powershell
.\scripts\docker\build-vlm-runtime.ps1
```

For VLM service-code changes, rebuild only the thin image:

```powershell
.\scripts\docker\build-vlm.ps1
```

The app and VLM intentionally use different Python contracts. Validate them with:

```powershell
python scripts/check_python_runtime_contract.py
python scripts/check_vlm_architecture.py
```
