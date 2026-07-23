param([switch]$NoCache)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$cacheArgs = @()
if ($NoCache) { $cacheArgs += "--no-cache" }
Push-Location $ProjectRoot
try {
    Write-Host "Building app runtime image: receipt-app-runtime:py311"
    docker build @cacheArgs -f docker/Dockerfile.app-runtime -t receipt-app-runtime:py311 .
} finally { Pop-Location }
