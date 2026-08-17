param(
    [switch]$Dev,
    [switch]$BuildAll,
    [switch]$BuildAppRuntime,
    [switch]$BuildApp,
    [switch]$NoCacheRuntime,
    [switch]$CleanModelCache,
    [string]$ModelCacheRoot = "..\..\model_cache"
)

$ErrorActionPreference = "Stop"

Write-Host "Starting Document Intelligence Pipeline..."
Write-Host "Assumption: Ollama is already running on the Windows host at http://localhost:11434"
Write-Host "Using shared model cache root: $ModelCacheRoot"
$env:MODEL_CACHE_ROOT = $ModelCacheRoot

if ($CleanModelCache) {
    Write-Host "Removing mounted Paddle and Hugging Face model caches..."
    Remove-Item -Recurse -Force (Join-Path $ModelCacheRoot "paddlex"), (Join-Path $ModelCacheRoot "huggingface") -ErrorAction SilentlyContinue
}

if ($BuildAll) {
    & "$PSScriptRoot\scripts\docker\build-all.ps1" -NoCacheRuntime:$NoCacheRuntime
} else {
    if ($BuildAppRuntime) {
        & "$PSScriptRoot\scripts\docker\build-app-runtime.ps1" -NoCache:$NoCacheRuntime
    }
    if ($BuildApp) {
        & "$PSScriptRoot\scripts\docker\build-app.ps1"
    }
}

if ($Dev) {
    docker compose -f docker-compose.yml -f docker-compose.dev.yml up -d --no-build
} else {
    docker compose up -d --no-build
}
