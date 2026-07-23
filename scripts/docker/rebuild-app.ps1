param([switch]$NoCacheRuntime)

$ErrorActionPreference = "Stop"
if ($NoCacheRuntime) { & "$PSScriptRoot\build-app-runtime.ps1" -NoCache }
else { & "$PSScriptRoot\build-app-runtime.ps1" }
& "$PSScriptRoot\build-app.ps1"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
Push-Location $ProjectRoot
try {
    docker compose -f docker-compose.yml -f docker-compose.dev.yml up -d `
      --no-build --force-recreate --no-deps receipt-app
} finally { Pop-Location }
