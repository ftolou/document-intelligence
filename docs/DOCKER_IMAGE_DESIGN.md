# Docker image design

The deployment contains one application image family:

```text
receipt-app-runtime:py311
  Python 3.11 and application dependencies

paddle-gemma-receipt-app:latest
  FROM receipt-app-runtime:py311
  repository source and static assets
```

Paddle text detection runs inside the application container. Qwen, Gemma, and embedding models run
through Ollama on the host. There is no standalone VLM image or service.

The production runtime installs only `requirements/app.txt`. Pytest, Ruff, mypy, coverage and
pre-commit remain development dependencies and are installed only in local/CI test environments.

Build dependencies only when `requirements/app.txt` or the runtime Dockerfile changes:

```powershell
.\scripts\docker\build-app-runtime.ps1
```

For normal source changes, rebuild only the thin application image:

```powershell
.\scripts\docker\build-app.ps1
```

The development override bind-mounts the repository and `var/` into `receipt-app`.
