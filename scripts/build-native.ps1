# Builds native/transcribe_shim.dll (transcribe.cpp + ggml, CPU backend).
# Needs: CMake, VS C++ tools, network (FetchContent).
param(
  [switch]$CPU,   # kept for callers; the shim is always CPU-only now
  [switch]$Stub   # build the shim without transcribe.cpp (no network)
)
$ErrorActionPreference = 'Stop'
$fetch = if ($Stub) { "OFF" } else { "ON" }
$args = @("-S", "native", "-B", "native/build", "-DHEX_FETCH_TRANSCRIBE=$fetch", "-DCMAKE_BUILD_TYPE=Release")
cmake @args
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
cmake --build native/build --config Release
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
Write-Host "built native/bin/transcribe_shim.dll"
