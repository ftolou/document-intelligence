$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
Push-Location $ProjectRoot
try { docker compose -f docker-compose.yml -f docker-compose.dev.yml restart receipt-vlm }
finally { Pop-Location }
