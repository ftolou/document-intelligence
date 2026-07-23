# Shared Model Cache Patch

The project was patched to use the existing shared model cache at:

```text
../../model_cache
```

## Changed files

- `docker-compose.yml`
  - Replaced project-local `./model_cache/...` volume mounts with `${MODEL_CACHE_ROOT:-../../model_cache}/...` for both `receipt-app` and `receipt-vlm`.
- `.env.example`
  - Set `MODEL_CACHE_ROOT=../../model_cache`.
- `start_windows.ps1`
  - Added `-ModelCacheRoot` parameter with default `..\..\model_cache`.
  - Exports `MODEL_CACHE_ROOT` before running Docker Compose.
- `README.md`
  - Added shared cache documentation and corrected cache-cleaning commands.

## Important

Ollama/Gemma currently runs on the Windows host, not in this Docker Compose stack. Therefore, this Docker Compose change controls the PaddleOCR/PaddleOCR-VL/Hugging Face/Torch/Paddle caches inside Docker. Gemma/Ollama model storage remains controlled by the host Ollama installation unless Ollama or vLLM is later containerized.
