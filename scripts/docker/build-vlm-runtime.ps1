param([switch]$NoCache)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$cacheArgs = @()
if ($NoCache) { $cacheArgs += "--no-cache" }
Push-Location $ProjectRoot
try {
    Write-Host "Building VLM runtime image: receipt-vlm-runtime:cu126"
    docker build @cacheArgs -f docker/Dockerfile.vlm-runtime-cu126 -t receipt-vlm-runtime:cu126 .
} finally { Pop-Location }
