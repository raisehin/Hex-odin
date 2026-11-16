# Explicit model provisioning for Hex. Not part of scripts/build-ci.ps1.
param(
    [switch]$CheckOnly,
    [string]$InstallDir = (Join-Path $env:APPDATA 'hex\models')
)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot

$phononName = 'phonon-2.5-en-Q8_0.gguf'
$phononBytes = 739508608
$phononSha = 'a7c32cae47157a9da95730963f5efa9d6e2f9bd4f7203887fe50ee4ce75aba5f'
$vadName = 'silero-vad-v6.2-F32.gguf'
$vadBytes = 1241056
$vadSha = '0d871a800c420fdd5d1b17a3aea4296d1c39608f0ff4a9295eb76457eb1c88e6'
$vadUrl = 'https://huggingface.co/handy-computer/silero-vad-v6.2-gguf/resolve/a9b2d61c20ae4ddf3644edb34969e8185a485448/silero-vad-v6.2-F32.gguf'

function Test-Installed([string]$name, [long]$bytes, [string]$sha) {
    $path = Join-Path $InstallDir $name
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
        Write-Host "$name : MISSING ($path)"
        return $false
    }
    $item = Get-Item -LiteralPath $path
    $actualSha = (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($item.Length -ne $bytes -or $actualSha -ne $sha) {
        Write-Host "$name : MISMATCH size=$($item.Length) expected=$bytes sha256=$actualSha expected=$sha"
        return $false
    }
    Write-Host "$name : OK size=$($item.Length) sha256=$actualSha"
    return $true
}

function Install-Verified([string]$source, [string]$name, [long]$bytes, [string]$sha) {
    $dest = Join-Path $InstallDir $name
    $tmp = Join-Path $InstallDir "$name.tmp-$([guid]::NewGuid().ToString('N'))"
    try {
        Copy-Item -LiteralPath $source -Destination $tmp -Force
        $item = Get-Item -LiteralPath $tmp
        $actualSha = (Get-FileHash -LiteralPath $tmp -Algorithm SHA256).Hash.ToLowerInvariant()
        if ($item.Length -ne $bytes -or $actualSha -ne $sha) {
            throw "$name staged file failed verification: size=$($item.Length) expected=$bytes sha256=$actualSha expected=$sha"
        }
        if (Test-Path -LiteralPath $dest -PathType Leaf) {
            $backup = Join-Path $InstallDir "$name.bak-$([guid]::NewGuid().ToString('N'))"
            [IO.File]::Replace($tmp, $dest, $backup, $true)
            Remove-Item -LiteralPath $backup -Force
        } else {
            Move-Item -LiteralPath $tmp -Destination $dest -Force
        }
        Write-Host "$name : INSTALLED $dest"
    } finally {
        if (Test-Path -LiteralPath $tmp -PathType Leaf) { Remove-Item -LiteralPath $tmp -Force }
    }
}

function Install-Vad {
    $tmp = Join-Path $InstallDir "$vadName.download-$([guid]::NewGuid().ToString('N'))"
    Write-Host "$vadName : downloading pinned GGUF from $vadUrl"
    try {
        & curl.exe -fL --progress-bar -o $tmp $vadUrl
        if ($LASTEXITCODE -ne 0) { throw "VAD download failed with curl exit $LASTEXITCODE" }
        Install-Verified $tmp $vadName $vadBytes $vadSha
    } finally {
        if (Test-Path -LiteralPath $tmp -PathType Leaf) { Remove-Item -LiteralPath $tmp -Force }
    }
}

New-Item -ItemType Directory -Force -Path $InstallDir | Out-Null
$vadOk = Test-Installed $vadName $vadBytes $vadSha
$phononOk = Test-Installed $phononName $phononBytes $phononSha

if ($CheckOnly) {
    if ($vadOk -and $phononOk) { exit 0 }
    exit 1
}

if (-not $vadOk) { Install-Vad }

if (-not $phononOk) {
    $script = Join-Path $root 'scripts\fetch-phonon25.ps1'
    Write-Host "$phononName : converting pinned Fermion archive via $script"
    & powershell.exe -NoProfile -ExecutionPolicy Bypass -File $script -Install -InstallDir $InstallDir
    if ($LASTEXITCODE -ne 0) { throw "Phonon provisioning failed with exit $LASTEXITCODE" }
}

$vadOk = Test-Installed $vadName $vadBytes $vadSha
$phononOk = Test-Installed $phononName $phononBytes $phononSha
if (-not ($vadOk -and $phononOk)) {
    Write-Error 'model provisioning failed verification'
    exit 1
}
Write-Host 'models: ready'
