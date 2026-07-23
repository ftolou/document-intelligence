param([string]$RuntimeImage = "receipt-vlm-runtime:cu126")

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
Push-Location $ProjectRoot
try {
    Write-Host "Building thin VLM image from $RuntimeImage"
    docker build -f docker/Dockerfile.vlm-app `
      --build-arg VLM_RUNTIME_IMAGE=$RuntimeImage `
      -t paddle-gemma-receipt-vlm:gpu-python-cu126 .
} finally { Pop-Location }
