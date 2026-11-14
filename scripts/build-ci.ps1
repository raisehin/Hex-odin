# Reproducible app-only build for a clean checkout and for CI.
#
#   .\scripts\build-ci.ps1
#   .\scripts\build-ci.ps1 -Odin C:\path\to\odin.exe
#
# This script is tracked so a clone has a build path without the maintainer's
# local build.ps1. It deliberately does not touch the native shim, model
# conversion, or any ambient icon tooling: the icon is the checked-in
# assets\hex.res generated from Hex.png.
param(
    [string]$Odin,
    [string]$Microarch = $(if ($env:HEX_MICROARCH) { $env:HEX_MICROARCH } else { 'x86-64' })
)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
Push-Location $root
try {
    function Resolve-Odin([string]$path) {
        if ([string]::IsNullOrWhiteSpace($path)) { return $null }
        if (Test-Path $path -PathType Leaf) { return (Resolve-Path $path).Path }
        $inside = Join-Path $path 'odin.exe'
        if (Test-Path $inside -PathType Leaf) { return (Resolve-Path $inside).Path }
        return $null
    }

    $odinExe = $null
    if ($Odin) {
        $odinExe = Resolve-Odin $Odin
        if (-not $odinExe) { throw "-Odin '$Odin' is neither odin.exe nor a directory containing it" }
    } elseif ($env:ODIN) {
        $odinExe = Resolve-Odin $env:ODIN
    }
    if (-not $odinExe) {
        $onPath = Get-Command odin -ErrorAction SilentlyContinue
        if ($onPath) { $odinExe = $onPath.Source }
    }
    if (-not $odinExe) {
        throw @"
Odin compiler not found.
  pass it:  .\scripts\build-ci.ps1 -Odin <path\to\odin.exe>
  or set:   `$env:ODIN = <path\to\odin.exe>   (or its directory)
  or add:   odin.exe to PATH
"@
    }

    $res = Join-Path $root 'assets\hex.res'
    if (-not (Test-Path $res -PathType Leaf)) {
        throw "missing checked-in icon resource: $res"
    }

    $bin = Join-Path $root 'bin'
    New-Item -ItemType Directory -Force -Path $bin | Out-Null
    $exe = Join-Path $bin 'hex.exe'

    # /Brepro removes the PE timestamp; source-code-locations:none keeps the
    # checkout path out of release binaries. Both are required for the
    # two-clean-build hash comparison in CI.
    $flags = @(
        '-o:speed',
        "-microarch:$Microarch",
        '-extra-linker-flags:/Brepro',
        '-source-code-locations:none',
        "-resource:$res"
    )

    & $odinExe build src -out:$exe @flags
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    Write-Host "built $exe"
} finally {
    Pop-Location
}
