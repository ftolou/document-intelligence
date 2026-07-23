param([switch]$NoCacheRuntime)

$ErrorActionPreference = "Stop"
if ($NoCacheRuntime) {
    & "$PSScriptRoot\build-app-runtime.ps1" -NoCache
    & "$PSScriptRoot\build-vlm-runtime.ps1" -NoCache
} else {
    & "$PSScriptRoot\build-app-runtime.ps1"
    & "$PSScriptRoot\build-vlm-runtime.ps1"
}
& "$PSScriptRoot\build-app.ps1"
& "$PSScriptRoot\build-vlm.ps1"
