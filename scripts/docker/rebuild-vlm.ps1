param([switch]$NoCacheRuntime)

$ErrorActionPreference = "Stop"
if ($NoCacheRuntime) { & "$PSScriptRoot\build-vlm-runtime.ps1" -NoCache }
else { & "$PSScriptRoot\build-vlm-runtime.ps1" }
& "$PSScriptRoot\build-vlm.ps1"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
Push-Location $ProjectRoot
try {
    docker compose -f docker-compose.yml -f docker-compose.dev.yml up -d `
      --no-build --force-recreate --no-deps receipt-vlm
} finally { Pop-Location }
