# Builds native/transcribe_shim.dll (transcribe.cpp + ggml, CPU backend).
# Needs: CMake, VS C++ tools, network (FetchContent) unless -Stub is used.
#
# The app build (build.ps1 -Release) is separate and does not need this DLL.
param(
  [switch]$CPU,   # kept for callers; the shim is always CPU-only now
  [switch]$Stub   # build the shim without transcribe.cpp (no network)
)
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$native = Join-Path $root 'native'
$build = Join-Path $native 'build'
$fetch = if ($Stub) { 'OFF' } else { 'ON' }
$manifest = Get-Content -Raw (Join-Path $root 'dependencies.json') | ConvertFrom-Json
$rev = $manifest.transcribe_cpp.revision
$cmake = Get-Command cmake -ErrorAction SilentlyContinue
if (-not $cmake) { throw 'build-native.ps1 needs cmake on PATH' }
$cmakeArgs = @(
  '-S', $native,
  '-B', $build,
  "-DHEX_FETCH_TRANSCRIBE=$fetch",
  "-DTRANSCRIBE_REV=$rev",
  '-DCMAKE_BUILD_TYPE=Release'
)
& $cmake.Source @cmakeArgs
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $cmake.Source --build $build --config Release
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
Write-Host "built native/bin/transcribe_shim.dll"
