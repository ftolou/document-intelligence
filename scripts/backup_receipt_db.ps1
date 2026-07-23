param(
    [string]$DatabasePath = "var\database\receipt_intelligence.db",
    [string]$BackupDirectory = "var\database\backups"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Push-Location $ProjectRoot
try {
    $Source = Join-Path $ProjectRoot $DatabasePath
    if (-not (Test-Path $Source)) {
        throw "Receipt database not found: $Source"
    }

    $DestinationDirectory = Join-Path $ProjectRoot $BackupDirectory
    New-Item -ItemType Directory -Force -Path $DestinationDirectory | Out-Null
    $Timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $Destination = Join-Path $DestinationDirectory "receipt_intelligence_$Timestamp.db"
    Copy-Item -Path $Source -Destination $Destination -Force
    Write-Host "Receipt database backup created: $Destination"
}
finally {
    Pop-Location
}
