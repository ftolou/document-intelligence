# VLM Troubleshooting

## `Place(undefined:0)` / `is_bfloat16_supported()` in Python API mode

Symptom:

```text
PaddleOCRVL run failed: TypeError: is_bfloat16_supported(): incompatible function arguments ... Invoked with: Place(undefined:0)
```

Meaning:

- The PaddleOCR-VL Python API is active.
- The model cache is not the primary problem.
- Paddle/PaddleX is calling Paddle's BF16 capability check with an undefined current device place.

Mitigation in this repository:

- `vl_engine.py` sets CUDA/Paddle environment variables before importing PaddleOCR-VL.
- It calls `paddle.set_device("gpu:0")`.
- It forces Paddle's expected place where supported.
- It installs a conservative BF16 guard that returns `False` instead of crashing on `Place(undefined:0)`.
- It imports `PaddleOCRVL` only after device initialization.
- The VLM service bind-mounts `./vl_engine.py` into `/app/vl_engine.py` so runner fixes can be tested without rebuilding the heavy image.

Recommended local test:

```powershell
docker compose down
docker compose up -d --no-build
Invoke-RestMethod http://localhost:7870/health | ConvertTo-Json -Depth 5
```

If the error persists, inspect the new `traceback_tail` and `runtime` fields in the VLM raw JSON.
