$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
Push-Location $projectRoot
try {
    python scripts/verify_runtime_layout.py
}
finally {
    Pop-Location
}
