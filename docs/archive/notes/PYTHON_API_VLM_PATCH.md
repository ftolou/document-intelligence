# Python API VLM patch

This patch switches the VLM baseline from CLI runner to PaddleOCR-VL Python API runner.

Changed files:

- `docker-compose.yml`
  - `receipt-vlm.image` changed to `paddle-gemma-receipt-vlm:gpu-python-cu126` to avoid accidentally reusing the old CLI image.
  - `VLM_SERVICE_RUNNER` changed from `cli` to `python`.
- `Dockerfile.vlm-service.gpu.heavy_cuda126`
  - default `VLM_SERVICE_RUNNER=python`.
- `vlm_service.py`
  - default runner changed to `python`.
  - service version changed to `v14_14_1_2_python_api_vlm_service`.
- `.env.example`
  - `VLM_SERVICE_RUNNER=python`.
- `README.md` and `start_windows.ps1`
  - updated commands to rebuild the VLM image and verify Python API mode.

Important: existing containers/images must be rebuilt. If you still see `runner: cli` or `service_version: v14_7_6_gpu_cli_vlm_service`, you are still running the old VLM image.

Use:

```powershell
docker compose down
docker compose build --no-cache receipt-vlm
docker compose up
```

If the run fails with `SafetensorError: incomplete metadata`, clear the mounted model cache:

```powershell
Remove-Item -Recurse -Force .\model_cache\paddlex, .\model_cache\huggingface -ErrorAction SilentlyContinue
docker compose up --build
```
