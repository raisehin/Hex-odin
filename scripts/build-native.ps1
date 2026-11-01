# Builds native/transcribe_shim.dll (transcribe.cpp + ggml + Vulkan shaders).
# Needs: CMake, VS C++ tools, network (FetchContent), Vulkan SDK for GPU.
param(
  [switch]$CPU,   # CPU-only ggml: no Vulkan SDK, slower, works everywhere
  [switch]$Stub   # build the shim without transcribe.cpp (no network)
)
$ErrorActionPreference = 'Stop'
$fetch = if ($Stub) { "OFF" } else { "ON" }
$args = @("-S", "native", "-B", "native/build", "-DHEX_FETCH_TRANSCRIBE=$fetch", "-DCMAKE_BUILD_TYPE=Release")
if ($CPU) { $args += "-DGGML_VULKAN=OFF" }
cmake @args
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
cmake --build native/build --config Release
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
Write-Host "built native/bin/transcribe_shim.dll"
