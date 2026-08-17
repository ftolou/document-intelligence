# Security boundaries

- Receipt-upload form fields cannot override the Ollama endpoint or transcription model.
- Uploaded filenames and artifact paths are sanitized and confined to managed runtime roots.
- Extraction model calls go through provider-neutral gateways and observed adapters.
- Paddle receives only the uploaded image path and server-managed detection settings.
- RAG-SQL accepts only validated read-only statements against curated views and functions.
- Batch paths are confined to the configured batch root unless explicitly allowed by server policy.
- Subprocess calls with `shell=True` are forbidden by repository checks.

Run `python scripts/check_security_boundaries.py` to verify these constraints.
