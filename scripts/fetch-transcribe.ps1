# Fetches the exact transcribe.cpp revision named in dependencies.json.
#
#   .\scripts\fetch-transcribe.ps1
#   .\scripts\fetch-transcribe.ps1 -Destination C:\src\transcribe.cpp
#
# This is for the optional native shim and for fetch-phonon25.ps1. The default
# app build (odin build src) does not need this checkout.
param(
    [string]$Destination = (Join-Path (Split-Path -Parent $PSScriptRoot) 'native\third_party\transcribe.cpp'),
    [switch]$Force
)

$ErrorActionPreference = 'Stop'

$manifestPath = Join-Path (Split-Path -Parent $PSScriptRoot) 'dependencies.json'
$manifest = Get-Content -Raw $manifestPath | ConvertFrom-Json
$repo = $manifest.transcribe_cpp.repository
$rev = $manifest.transcribe_cpp.revision

if (Test-Path (Join-Path $Destination '.git')) {
    $current = (& git -C $Destination rev-parse HEAD).Trim()
    if ($current -eq $rev) {
        Write-Host "transcribe.cpp already at $rev"
        return
    }
    & git -C $Destination fetch --depth 1 origin $rev
    if ($LASTEXITCODE -ne 0) { throw "git fetch failed for $Destination" }
    & git -C $Destination checkout --detach $rev
    if ($LASTEXITCODE -ne 0) { throw "git checkout failed for $Destination" }
} elseif (Test-Path $Destination) {
    if (-not $Force) {
        throw "$Destination exists but is not a git checkout; move it or pass -Force"
    }
    Remove-Item -Recurse -Force $Destination
    & git clone $repo $Destination
    if ($LASTEXITCODE -ne 0) { throw "git clone failed for $repo" }
    & git -C $Destination checkout --detach $rev
    if ($LASTEXITCODE -ne 0) { throw "git checkout failed for $Destination" }
} else {
    & git clone $repo $Destination
    if ($LASTEXITCODE -ne 0) { throw "git clone failed for $repo" }
    & git -C $Destination checkout --detach $rev
    if ($LASTEXITCODE -ne 0) { throw "git checkout failed for $Destination" }
}

$actual = (& git -C $Destination rev-parse HEAD).Trim()
if ($actual -ne $rev) {
    throw "transcribe.cpp is at $actual, expected $rev"
}
Write-Host "transcribe.cpp $rev -> $Destination"
