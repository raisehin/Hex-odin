# Builds hex.exe with Odin, then the transcribe shim and copies it next to the exe.
#
#   .\build.ps1 -Release              # Odin app only (~5 s)
#   .\build.ps1 -Release -Native      # also (re)build native/transcribe_shim (cmake)
param([switch]$Release, [switch]$Native)

$ErrorActionPreference = 'Stop'
$odin  = "C:\Users\rhais\Documents\odin-windows-amd64-dev-2026-10\dist\odin.exe"
$flags = @(if ($Release) { "-o:speed" } else { "-debug" })

# App icon: Hex.png -> hex.ico + hex.rc (scripts/make-icon.py) -> hex.res (rc.exe)
# -> linked into the exe. The .res is what makes Explorer, the taskbar button and
# Alt-Tab show the mark, and it is regenerated only when the art or .ico changed.
$res = "hex.res"
$stale = -not (Test-Path $res)
if (-not $stale) {
    $resTime = (Get-Item $res).LastWriteTimeUtc
    foreach ($src in @("Hex.png", "hex.ico", "hex.rc")) {
        if ((Test-Path $src) -and (Get-Item $src).LastWriteTimeUtc -gt $resTime) { $stale = $true }
    }
}
if ($stale) {
    $rc = Get-ChildItem "${env:ProgramFiles(x86)}\Windows Kits\10\bin\*\x64\rc.exe" -ErrorAction SilentlyContinue |
        Sort-Object FullName -Descending | Select-Object -First 1
    if ($rc -and (Get-Command python -ErrorAction SilentlyContinue)) {
        & python scripts/make-icon.py
        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
        & $rc.FullName /nologo /fo $res hex.rc
        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
        Write-Host "rebuilt $res from Hex.png"
    } elseif (Test-Path $res) {
        Write-Warning "rc.exe or python missing; keeping the existing $res (icon may be stale)"
    } else {
        Write-Warning "rc.exe or python missing and no $res; building without an embedded icon"
    }
}

$linkArgs = @()
if (Test-Path $res) { $linkArgs += "-resource:$res" }

& $odin build src -out:hex.exe @linkArgs @flags
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

if ($Native) {
    cmake -S native -B native/build -DCMAKE_BUILD_TYPE=Release
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    cmake --build native/build --config Release
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

if (Test-Path "native/bin/transcribe_shim.dll") {
    Copy-Item "native/bin/transcribe_shim.dll" "transcribe_shim.dll" -Force
    Write-Host "copied transcribe_shim.dll next to hex.exe"
}
Write-Host "built hex.exe"
