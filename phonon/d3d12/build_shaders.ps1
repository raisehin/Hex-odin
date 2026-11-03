# Compiles the D3D12 compute shaders used by the phonon GPU encoder path.
#
# Requires dxc.exe. The Windows SDK ships one (no Vulkan SDK or downloads):
#   C:\Program Files (x86)\Windows Kits\10\bin\10.0.19041.0\x64\dxc.exe
#
# Usage: powershell -ExecutionPolicy Bypass -File phonon\d3d12\build_shaders.ps1
#
# The .dxil outputs are #load-ed by the Odin package, so rerun this whenever a
# .hlsl file changes.

$ErrorActionPreference = "Stop"
$here = Split-Path -Parent $MyInvocation.MyCommand.Path

function Find-Dxc {
    if ($env:DXC) { return $env:DXC }
    $roots = @(
        (Join-Path ${env:ProgramFiles(x86)} "Windows Kits\10\bin"),
        (Join-Path $env:ProgramFiles "Windows Kits\10\bin")
    )
    foreach ($root in $roots) {
        if (-not (Test-Path $root)) { continue }
        $cand = Get-ChildItem -Path $root -Filter dxc.exe -Recurse -ErrorAction SilentlyContinue |
            Where-Object { $_.FullName -match '\\x64\\dxc\.exe$' } |
            Sort-Object FullName -Descending |
            Select-Object -First 1
        if ($cand) { return $cand.FullName }
    }
    return $null
}

$dxc = Find-Dxc
if (-not $dxc) { throw "dxc.exe not found; set DXC to its full path" }
Write-Host "dxc: $dxc"

$shaders = @(
    @{ Src = "q8_0_gemm.hlsl"; Out = "q8_0_gemm.dxil"; Entry = "main"; Profile = "cs_6_4" },
    @{ Src = "flops.hlsl";     Out = "flops.dxil";     Entry = "main"; Profile = "cs_6_4" }
)

foreach ($s in $shaders) {
    $src = Join-Path $here $s.Src
    $dst = Join-Path $here $s.Out
    & $dxc -T $s.Profile -E $s.Entry -O3 -Fo $dst $src
    if ($LASTEXITCODE -ne 0) { throw "dxc failed for $($s.Src)" }
    Write-Host ("built {0} ({1} bytes)" -f $s.Out, (Get-Item $dst).Length)
}
