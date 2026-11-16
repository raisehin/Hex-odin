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
    [string]$Quant = "Q8_0",
    [string]$InstallDir = (Join-Path $env:APPDATA "hex\models")
)

$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent $PSScriptRoot
$src = Join-Path $repoRoot "native\third_party\transcribe.cpp"
$slug = if ($Variant -eq "multi") { "phonon-2.5" } else { "phonon-2.5-en" }
$outDir = Join-Path $repoRoot "native\models\$slug"
$f32 = Join-Path $outDir "$slug-F32.gguf"

if (-not (Test-Path (Join-Path $src "scripts\convert-parakeet.py"))) {
    Write-Host "transcribe.cpp checkout missing at $src; fetching pinned revision..." -ForegroundColor Cyan
    & (Join-Path $PSScriptRoot "fetch-transcribe.ps1") -Destination $src
    if ($LASTEXITCODE -ne 0) { throw "fetch-transcribe.ps1 failed" }
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
    python (Join-Path $PSScriptRoot "convert-phonon25.py") --variant $Variant
    if ($LASTEXITCODE -ne 0) { throw "convert-phonon25.py failed" }
}

# 3. Quantize with transcribe.cpp's own tool so the format matches the engine.
$quantize = Join-Path $repoRoot "native\build-transcribe\bin\Release\transcribe-quantize.exe"
if (-not (Test-Path $quantize)) {
    Write-Host "building transcribe-quantize..." -ForegroundColor Cyan
    $buildDir = Join-Path $repoRoot "native\build-transcribe"
    cmake -S $src -B $buildDir -DCMAKE_BUILD_TYPE=Release `
          -DGGML_VULKAN=OFF -DTRANSCRIBE_BUILD_TOOLS=ON | Out-Null
    cmake --build $buildDir --config Release --target transcribe-quantize | Out-Null
}

$out = Join-Path $outDir ("$slug-" + $Quant + ".gguf")
Write-Host "quantizing -> $out" -ForegroundColor Cyan
& $quantize $f32 $out --quant $Quant
if ($LASTEXITCODE -ne 0) { throw "transcribe-quantize failed" }

if ($Install) {
    New-Item -ItemType Directory -Force -Path $InstallDir | Out-Null
    $expected = Get-Content -Raw (Join-Path $root 'scripts\models_extra.json') | ConvertFrom-Json |
        Where-Object { $_.id -eq 'phonon25_en' } | Select-Object -First 1
    if (-not $expected) { throw 'phonon25_en is missing from scripts/models_extra.json' }
    $item = Get-Item -LiteralPath $out
    $actualSha = (Get-FileHash -LiteralPath $out -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($item.Length -ne $expected.bytes -or $actualSha -ne $expected.sha256) {
        throw "quantized model failed verification: size=$($item.Length) expected=$($expected.bytes) sha256=$actualSha expected=$($expected.sha256)"
    }
    $dest = Join-Path $InstallDir (Split-Path $out -Leaf)
    $tmp = Join-Path $InstallDir "$($expected.filename).tmp-$([guid]::NewGuid().ToString('N'))"
    try {
        Copy-Item -LiteralPath $out -Destination $tmp -Force
        if (Test-Path -LiteralPath $dest -PathType Leaf) {
            $backup = Join-Path $InstallDir "$($expected.filename).bak-$([guid]::NewGuid().ToString('N'))"
            [IO.File]::Replace($tmp, $dest, $backup, $true)
            Remove-Item -LiteralPath $backup -Force
        } else {
            Move-Item -LiteralPath $tmp -Destination $dest -Force
        }
    } finally {
        if (Test-Path -LiteralPath $tmp -PathType Leaf) { Remove-Item -LiteralPath $tmp -Force }
    }
    Write-Host "verified and installed into $InstallDir" -ForegroundColor Green
    Write-Host "run: .\bin\hex.exe models list | findstr phonon"
}

Write-Host "done: $out" -ForegroundColor Green
