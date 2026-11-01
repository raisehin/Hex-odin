# Creates Desktop + Start Menu shortcuts for hex.exe, and a dot-scope icon.
#
#   .\scripts\install-shortcut.ps1            # Desktop + Start Menu
#   .\scripts\install-shortcut.ps1 -Startup   # ...and launch at login
param([switch]$Startup)

$ErrorActionPreference = 'Stop'
$exe = (Resolve-Path "$PSScriptRoot\..\hex.exe").Path
$dir = Split-Path $exe
# Shortcuts point at hex.ico, which scripts/make-icon.py builds from Hex.png and
# build.ps1 links into the exe. Only regenerate it here if it went missing.
$icon = Join-Path $dir "hex.ico"
if (-not (Test-Path $icon)) {
    & python (Join-Path $PSScriptRoot "make-icon.py")
    if ($LASTEXITCODE -ne 0) { throw "could not build $icon from Hex.png" }
}

$shell = New-Object -ComObject WScript.Shell
function New-HexShortcut($path) {
    $sc = $shell.CreateShortcut($path)
    $sc.TargetPath = $exe
    $sc.WorkingDirectory = $dir
    $sc.IconLocation = "$icon,0"
    $sc.Description = "Hex - local dictation (Ctrl+Space)"
    $sc.Save()
    Write-Host "created $path"
}

New-HexShortcut (Join-Path ([Environment]::GetFolderPath('Desktop')) "Hex.lnk")
New-HexShortcut (Join-Path ([Environment]::GetFolderPath('Programs')) "Hex.lnk")

if ($Startup) {
    New-HexShortcut (Join-Path ([Environment]::GetFolderPath('Startup')) "Hex.lnk")
}

Write-Host "hex.exe with no arguments starts the dictation host." -ForegroundColor Green
