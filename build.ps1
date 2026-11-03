# Builds bin\hex.exe with Odin, then the transcribe shim next to it.
#
#   .\build.ps1 -Release              # Odin app only (~5 s)  -> bin\hex.exe
#   .\build.ps1 -Release -Native      # also (re)build native/transcribe_shim (cmake)
#   .\build.ps1 -Release -Odin C:\path\to\odin.exe
#
# Nothing here is tied to this machine: the compiler is found from -Odin,
# $env:ODIN, PATH, then the usual install locations, and every path below is
# resolved from this script's own directory, so it runs from any working dir.
param(
    [switch]$Release,
    [switch]$Native,
    [string]$Odin
)

$ErrorActionPreference = 'Stop'

$root = $PSScriptRoot
if (-not $root) { $root = Split-Path -Parent $MyInvocation.MyCommand.Path }
Push-Location $root

# --- tool discovery ---------------------------------------------------------
# Every lookup returns a path or $null; only the compiler is required, and its
# absence is reported with the ways to point at it.

function Resolve-Tool([string]$path) {
    if ([string]::IsNullOrWhiteSpace($path)) { return $null }
    if (Test-Path $path -PathType Leaf) { return (Resolve-Path $path).Path }
    $inside = Join-Path $path 'odin.exe'
    if (Test-Path $inside -PathType Leaf) { return (Resolve-Path $inside).Path }
    return $null
}

function Find-Odin {
    if ($Odin) {
        $hit = Resolve-Tool $Odin
        if (-not $hit) { throw "-Odin '$Odin' is neither odin.exe nor a directory containing it" }
        return $hit
    }
    if ($env:ODIN) {
        $hit = Resolve-Tool $env:ODIN
        if ($hit) { return $hit }
    }
    $onPath = Get-Command odin -ErrorAction SilentlyContinue
    if ($onPath) { return $onPath.Source }

    # Release archives unpack as <name>\dist\odin.exe; scoops and hand-copies
    # put it in bin\ or at the top level. One directory level, no deep scans.
    $parents = @(
        $env:LOCALAPPDATA, $env:APPDATA, $env:ProgramFiles, $HOME,
        (Join-Path $HOME 'Downloads'), (Join-Path $HOME 'Documents'),
        (Join-Path $HOME 'scoop\apps'), (Join-Path $HOME 'tools'), 'C:\tools', 'C:\dev'
    ) | Where-Object { $_ -and (Test-Path $_) }
    foreach ($parent in $parents) {
        $dirs = Get-ChildItem -Path $parent -Directory -Filter 'odin*' -ErrorAction SilentlyContinue
        if ($parent -like '*scoop\apps*') {
            $dirs = Get-ChildItem -Path (Join-Path $parent 'odin*') -Directory -ErrorAction SilentlyContinue
        }
        foreach ($dir in @($dirs)) {
            foreach ($sub in @('dist', 'bin', '')) {
                $cand = if ($sub) { Join-Path $dir.FullName (Join-Path $sub 'odin.exe') } else { Join-Path $dir.FullName 'odin.exe' }
                if (Test-Path $cand -PathType Leaf) { return (Resolve-Path $cand).Path }
            }
        }
    }
    throw @"
Odin compiler not found.
  pass it:      .\build.ps1 -Odin <path\to\odin.exe>
  or set:       `$env:ODIN = <path\to\odin.exe>   (or its directory)
  or add it:    odin.exe on PATH
"@
}

function Find-Rc {
    $onPath = Get-Command rc.exe -ErrorAction SilentlyContinue
    if ($onPath) { return $onPath.Source }
    $kits = @("${env:ProgramFiles(x86)}\Windows Kits\10\bin", "$env:ProgramFiles\Windows Kits\10\bin") |
        Where-Object { $_ -and (Test-Path $_) }
    foreach ($kit in $kits) {
        $hit = Get-ChildItem -Path $kit -Filter rc.exe -Recurse -ErrorAction SilentlyContinue |
            Where-Object { $_.FullName -match '\\x64\\rc\.exe$' } |
            Sort-Object FullName -Descending | Select-Object -First 1
        if ($hit) { return $hit.FullName }
    }
    return $null
}

function Find-Python {
    foreach ($name in @('python', 'python3', 'py')) {
        $cmd = Get-Command $name -ErrorAction SilentlyContinue
        if ($cmd) {
            $pre = if ($name -eq 'py') { @('-3') } else { @() }
            return @{ exe = $cmd.Source; pre = $pre }
        }
    }
    return $null
}

# --- output -----------------------------------------------------------------
# Everything the build emits goes to bin\, which .gitignore covers: the exe, its
# PDB, the icon intermediates and the shim DLL the app loads from its own dir.
$bin = Join-Path $root 'bin'
New-Item -ItemType Directory -Force -Path $bin | Out-Null
$exe = Join-Path $bin 'hex.exe'
$res = Join-Path $bin 'hex.res'

try {
    $odinExe = Find-Odin
    Write-Host "odin: $odinExe"

    # -microarch:native: the built-in phonon engine's hot kernels are annotated
    # with their own target features, but the loops around them (mel, norms,
    # softmax) only vectorise when the build targets the host. Odin's default is
    # x86-64/SSE2; the encoder is 444 ms at that target against 326 ms at znver2.
    $flags = @(if ($Release) { "-o:speed" } else { "-debug" }) + "-microarch:native"

    # App icon: Hex.png -> bin\hex.ico + bin\hex.rc (scripts/make-icon.py) ->
    # bin\hex.res (rc.exe) -> linked into the exe. The .res is what makes
    # Explorer, the taskbar button and Alt-Tab show the mark, and it is rebuilt
    # only when the art or the intermediates changed.
    $stale = -not (Test-Path $res)
    if (-not $stale) {
        $resTime = (Get-Item $res).LastWriteTimeUtc
        foreach ($src in @("Hex.png", (Join-Path $bin 'hex.ico'), (Join-Path $bin 'hex.rc'))) {
            if ((Test-Path $src) -and (Get-Item $src).LastWriteTimeUtc -gt $resTime) { $stale = $true }
        }
    }
    if ($stale) {
        $rc = Find-Rc
        $py = Find-Python
        if ($rc -and $py) {
            $pyArgs = @($py.pre) + @((Join-Path $root 'scripts\make-icon.py'))
            & $py.exe @pyArgs
            if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
            # rc resolves the .rc's "1 ICON hex.ico" against the current
            # directory, so compile from bin\ where the pair lives.
            Push-Location $bin
            try {
                & $rc /nologo /fo 'hex.res' 'hex.rc'
                if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
            } finally {
                Pop-Location
            }
            Write-Host "rebuilt bin\hex.res from Hex.png"
        } elseif (Test-Path $res) {
            Write-Warning "rc.exe or python missing; keeping the existing bin\hex.res (icon may be stale)"
        } else {
            Write-Warning "rc.exe or python missing and no bin\hex.res; building without an embedded icon"
        }
    }

    $linkArgs = @()
    if (Test-Path $res) { $linkArgs += "-resource:$res" }

    & $odinExe build src -out:$exe @linkArgs @flags
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

    if ($Native) {
        $cmake = Get-Command cmake -ErrorAction SilentlyContinue
        if (-not $cmake) { throw "-Native needs cmake on PATH" }
        & $cmake.Source -S native -B native/build -DCMAKE_BUILD_TYPE=Release
        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
        & $cmake.Source --build native/build --config Release
        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    }

    # The app loads the shim from its own directory first (src/transcribe.odin).
    $shim = @(
        'native/bin/transcribe_shim.dll',
        'native/build/Release/transcribe_shim.dll',
        'native/build/transcribe_shim.dll'
    ) | Where-Object { Test-Path $_ } | Select-Object -First 1
    if ($shim) {
        Copy-Item $shim (Join-Path $bin 'transcribe_shim.dll') -Force
        Write-Host "copied transcribe_shim.dll next to the exe"
    }
    Write-Host "built $exe"
} finally {
    Pop-Location
}
