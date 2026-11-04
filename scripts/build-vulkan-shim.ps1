<#
.SYNOPSIS
    Builds native/bin/transcribe_shim.dll with ggml's Vulkan backend compiled in.

.DESCRIPTION
    The default shim (scripts/build-native.ps1) links ggml-cpu only, so ggml
    picks the CPU backend. This script builds the same shim against a Vulkan
    enabled ggml tree, which lets transcribe.cpp place the parakeet encoder and
    TDT decoder on the GPU.

    Measured on a Radeon 680M (RDNA2) with Phonon-2.5 en Q8_0, 10.87 s clip:
        CPU backend     2240 ms   (4.85x realtime)
        Vulkan backend   967 ms   (11.2x realtime)

    Requires a Vulkan SDK. Only three things are taken from it:
        $VulkanSdk\Bin\glslc.exe              compiles ggml's 155 .comp shaders
        $VulkanSdk\Include\vulkan\*.h         headers for ggml-vulkan
        $VulkanSdk\Lib\vulkan-1.lib           import library for the loader
    The Vulkan *runtime* (C:\Windows\System32\vulkan-1.dll) is separate and is
    what actually executes at run time.

.PARAMETER VulkanSdk
    Directory containing Bin, Include and Lib. Defaults to $env:VULKAN_SDK.

.PARAMETER BuildDir
    CMake tree for the Vulkan enabled ggml + transcribe build.

.EXAMPLE
    $env:VULKAN_SDK = "C:\VulkanSDK\1.4.309.0"
    .\scripts\build-vulkan-shim.ps1
#>
param(
    [string]$VulkanSdk = $env:VULKAN_SDK,
    [string]$BuildDir = "native/build-vk",
    [string]$Config = "Release"
)

$ErrorActionPreference = 'Stop'

$repo = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
Push-Location $repo
try {
    if (-not $VulkanSdk) {
        throw "No Vulkan SDK. Set VULKAN_SDK or pass -VulkanSdk. It must provide Bin\glslc.exe, Include\vulkan and Lib\vulkan-1.lib."
    }
    $VulkanSdk = (Resolve-Path $VulkanSdk).Path
    foreach ($f in 'Bin\glslc.exe', 'Include\vulkan\vulkan_core.h', 'Lib\vulkan-1.lib') {
        if (-not (Test-Path (Join-Path $VulkanSdk $f))) {
            throw "Vulkan SDK at $VulkanSdk is missing $f"
        }
    }
    # ggml-vulkan's CMake does find_package(Vulkan COMPONENTS glslc) and this is
    # the only place it looks.
    $env:VULKAN_SDK = $VulkanSdk
    Write-Host "Vulkan SDK: $VulkanSdk"

    $src = "native/third_party/transcribe.cpp"
    if (-not (Test-Path $src)) { throw "$src is missing; run scripts/build-native.ps1 once to fetch it" }

    # SPIRV-Headers ships a CMake package inside the SDK, but CMake does not
    # search $VULKAN_SDK/Lib/cmake for standalone packages on its own, so the
    # config directory has to be named explicitly.
    $spirvDir = Join-Path $VulkanSdk 'Lib\cmake'
    $configure = @(
        '-S', $src, '-B', $BuildDir,
        '-G', 'Visual Studio 17 2022',
        '-DCMAKE_BUILD_TYPE=Release',
        '-DBUILD_SHARED_LIBS=OFF',
        '-DGGML_NATIVE=ON',
        '-DGGML_CPU=ON',
        '-DGGML_VULKAN=ON',
        '-DGGML_BUILD_TESTS=OFF',
        '-DGGML_BUILD_EXAMPLES=OFF',
        '-DTRANSCRIBE_VULKAN=ON',
        '-DTRANSCRIBE_BUILD_TESTS=OFF',
        '-DTRANSCRIBE_BUILD_EXAMPLES=OFF',
        '-DTRANSCRIBE_BUILD_TOOLS=OFF',
        '-DTRANSCRIBE_BUILD_SHARED=OFF',
        "-DSPIRV-Headers_DIR=$spirvDir",
        "-DCMAKE_PREFIX_PATH=$VulkanSdk"
    )
    Write-Host "configuring $BuildDir"
    & cmake @configure
    if ($LASTEXITCODE -ne 0) { throw "cmake configure failed" }

    # Compiling the 155 shaders through glslc is the slow part (~4 min on 4 cores).
    Write-Host "building ggml-vulkan (compiles the shaders)"
    & cmake --build $BuildDir --config $Config
    if ($LASTEXITCODE -ne 0) { throw "cmake build failed" }

    # The shim is not part of that tree, so compile it against the libs above.
    # /MD is required: cl defaults to /MT, and mixing that with CMake's /MD
    # libraries produces a wall of __imp_ unresolved-crt symbols at link time.
    $vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
    $vcvars = $null
    if (Test-Path $vswhere) {
        $vsPath = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
        if ($vsPath) { $vcvars = Join-Path $vsPath 'VC\Auxiliary\Build\vcvars64.bat' }
    }
    if (-not $vcvars -or -not (Test-Path $vcvars)) { throw "vcvars64.bat not found; install the VS C++ build tools" }

    $out = Join-Path $repo 'native/bin'
    New-Item -ItemType Directory -Force $out | Out-Null
    $libs = @(
        "$repo\$BuildDir\src\$Config",
        "$repo\$BuildDir\ggml\src\$Config",
        "$repo\$BuildDir\ggml\src\ggml-vulkan\$Config"
    )
    $libArgs = ($libs | ForEach-Object { "/LIBPATH:`"$_`"" }) -join ' '
    $deploy = 'transcribe.lib ggml-vulkan.lib ggml.lib ggml-cpu.lib ggml-base.lib vulkan-1.lib advapi32.lib kernel32.lib user32.lib gdi32.lib winspool.lib shell32.lib ole32.lib oleaut32.lib uuid.lib comdlg32.lib'
    $cl = @(
        "call `"$vcvars`" >nul 2>&1 && cl /nologo /LD /O2 /MD /std:c++17 /EHsc",
        "/D WIN32 /D _WINDOWS /D NDEBUG /D HEX_HAVE_TRANSCRIBE=1 /D TRANSCRIBE_STATIC=1",
        "/I `"$repo\native\third_party\transcribe.cpp\include`"",
        "/I `"$repo\native\third_party\transcribe.cpp\ggml\include`"",
        "`"$repo\native\transcribe_shim.cpp`"",
        "/Fo:`"$out\\`" /Fe:`"$out\transcribe_shim.dll`"",
        "/link $libArgs /LIBPATH:`"$VulkanSdk\Lib`" $deploy"
    ) -join ' '
    Write-Host "linking native/bin/transcribe_shim.dll"
    cmd /c $cl
    if ($LASTEXITCODE -ne 0) { throw "shim link failed" }

    $dll = Get-Item (Join-Path $out 'transcribe_shim.dll')
    Write-Host ("built {0} ({1:N1} MB)" -f $dll.FullName, ($dll.Length / 1MB))
    Write-Host "copy it next to hex.exe, or run .\build.ps1 -Release which copies native/bin/ forwards"
}
finally {
    Pop-Location
}
