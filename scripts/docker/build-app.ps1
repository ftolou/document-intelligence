param([string]$RuntimeImage = "receipt-app-runtime:py311")

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
Push-Location $ProjectRoot
try {
    Write-Host "Building thin app image from $RuntimeImage"
    docker build -f docker/Dockerfile.app `
      --build-arg APP_RUNTIME_IMAGE=$RuntimeImage `
      -t paddle-gemma-receipt-app:latest .
} finally { Pop-Location }
