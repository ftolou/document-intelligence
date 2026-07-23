$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
Push-Location $ProjectRoot
try { docker compose -f docker-compose.yml -f docker-compose.dev.yml restart receipt-app receipt-vlm }
finally { Pop-Location }
