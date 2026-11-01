# Build Phonon-2.5 English for hex.
#
# Fermion Research ships Phonon-2.5 as a custom packed container
# (`phonon-2.5-en.bps.tar.zst`, ~164 MB for a 0.6B TDT), not GGUF, and public
# GGUF conversions use CrispASR's NeMo tensor names, which transcribe.cpp's
# loader rejects. Convert once:
#
#   upstream packed weights -> dense F32 GGUF (convert-phonon25.py) -> Q8_0
#
# Torch is used only while the container is expanded; loading, dictation and
# the GUI are Odin + C++. Run `pip uninstall torch` afterwards if you want it
# gone again.
#
#   .\scripts\fetch-phonon25.ps1            # convert into native/models/phonon-2.5-en
#   .\scripts\fetch-phonon25.ps1 -Install   # ...and drop the quant into %APPDATA%\hex\models
param(
    [switch]$Install,
    [ValidateSet("en", "multi")]
    [string]$Variant = "en",
    [string]$Quant = "Q8_0"
)

$ErrorActionPreference = 'Stop'
$repoRoot = "$PSScriptRoot\.."
$src = "$repoRoot\native\third_party\transcribe.cpp"
$slug = if ($Variant -eq "multi") { "phonon-2.5" } else { "phonon-2.5-en" }
$outDir = "$repoRoot\native\models\$slug"
$f32 = Join-Path $outDir "$slug-F32.gguf"

if (-not (Test-Path "$src\scripts\convert-parakeet.py")) {
    throw "transcribe.cpp checkout missing at $src (clone it first)"
}

# 1. Converter dependencies.
$haveTorch = $false
try { python -c "import torch" 2>$null; $haveTorch = $true } catch { }
if (-not $haveTorch) {
    Write-Host "installing converter deps (torch CPU, ~200 MB, one time)..." -ForegroundColor Cyan
    python -m pip install --quiet torch --index-url https://download.pytorch.org/whl/cpu
}
python -m pip install --quiet omegaconf zstandard huggingface_hub gguf

# 2. Convert the pinned upstream container to the canonical F32 GGUF.
if (-not (Test-Path $f32)) {
    New-Item -ItemType Directory -Force -Path $outDir | Out-Null
    Write-Host "converting Phonon-2.5 ($Variant) -> $f32" -ForegroundColor Cyan
    $env:PYTHONUTF8 = "1"
    python "$repoRoot\scripts\convert-phonon25.py" --variant $Variant
    if ($LASTEXITCODE -ne 0) { throw "convert-phonon25.py failed" }
}

# 3. Quantize with transcribe.cpp's own tool so the format matches the engine.
$quantize = "$repoRoot\native\build-transcribe\bin\Release\transcribe-quantize.exe"
if (-not (Test-Path $quantize)) {
    Write-Host "building transcribe-quantize..." -ForegroundColor Cyan
    cmake -S "$src" -B "$repoRoot\native\build-transcribe" -DCMAKE_BUILD_TYPE=Release `
          -DGGML_VULKAN=OFF -DTRANSCRIBE_BUILD_TOOLS=ON | Out-Null
    cmake --build "$repoRoot\native\build-transcribe" --config Release --target transcribe-quantize | Out-Null
}

$out = Join-Path $outDir ("$slug-" + $Quant + ".gguf")
Write-Host "quantizing -> $out" -ForegroundColor Cyan
& $quantize $f32 $out --quant $Quant
if ($LASTEXITCODE -ne 0) { throw "transcribe-quantize failed" }

if ($Install) {
    $models = Join-Path $env:APPDATA "hex\models"
    New-Item -ItemType Directory -Force -Path $models | Out-Null
    Copy-Item $out (Join-Path $models (Split-Path $out -Leaf)) -Force
    Write-Host "installed into $models" -ForegroundColor Green
    Write-Host "run: .\hex.exe models list | findstr phonon"
}

Write-Host "done: $out" -ForegroundColor Green
