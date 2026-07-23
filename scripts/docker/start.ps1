param([string]$ModelCacheRoot = "..\..\model_cache")

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$env:MODEL_CACHE_ROOT = $ModelCacheRoot
Push-Location $ProjectRoot
try {
    docker compose -f docker-compose.yml -f docker-compose.dev.yml up -d --no-build
} finally { Pop-Location }
