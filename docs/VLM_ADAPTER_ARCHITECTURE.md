# VLM service architecture

The receipt application and PaddleOCR-VL service are separate processes with
independent Python and dependency baselines.

```text
receipt-app (Python 3.11)
    RemoteVlmClient
          |
          | HTTP/JSON + shared image path
          v
receipt-vlm (Python 3.10)
    receipt_vlm_service
          |
          v
    paddleocr doc_parser CLI
```

## Boundary rules

- `receipt-app` never imports PaddleOCR, PaddleX, Torch, or the VLM CLI adapter.
- `receipt-vlm` never imports `receipt_intelligence`, LLM, RAG, storage, review,
  or query-observability modules.
- The VLM image copies only `services/receipt-vlm/src`.
- The development override mounts only that service source directory, not the
  complete repository.
- The API contract is `GET /health` and `POST /api/vlm/analyze`.
- Runtime configuration is server-owned. Request payloads cannot choose the
  command, runner, timeout, device, engine, or resize limit.

## Runtime contracts

The main application uses Python 3.11 and Ruff `py311`. The known-working CUDA
12.6 VLM runtime remains on Ubuntu 22.04 Python 3.10 and uses its own Ruff
`py310` configuration in `services/receipt-vlm/pyproject.toml`.

Run:

```powershell
python scripts/check_python_runtime_contract.py
python scripts/check_vlm_architecture.py
```
