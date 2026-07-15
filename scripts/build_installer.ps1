# Builds WhisperFlow-Setup.exe end to end:
#   1. PyInstaller freeze  -> dist\WhisperFlow\WhisperFlow.exe (or WhisperFlow-Cloud\WhisperFlow-Cloud.exe)
#   2. Inno Setup compile  -> installer\Output\WhisperFlow-Setup.exe
#
# One-time prerequisite on the build machine:
#   Inno Setup 6 — https://jrsoftware.org/isinfo.php
# The script creates/updates its own clean build venv (.venv-build) so the
# frozen exe contains ONLY the app's real dependencies — building from a
# global Python that has other packages installed produces a multi-GB dist.
#
# Usage:
#   powershell -ExecutionPolicy Bypass -File scripts\build_installer.ps1              # default: slim cloud-base installer
#   powershell -ExecutionPolicy Bypass -File scripts\build_installer.ps1 -Full        # full (~1GB) installer, local inference bundled
#   powershell -ExecutionPolicy Bypass -File scripts\build_installer.ps1 -LocalPack   # zip the local-inference pack for a GitHub release

param(
    [switch]$Full,       # build the full (current, ~1GB) installer instead of the slim cloud one
    [switch]$LocalPack   # build+zip the local-inference pack for a GitHub release, instead of the installer
)

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo

Write-Output "== 0/2 Build venv =="
$python = "$repo\.venv-build\Scripts\python.exe"
if (-not (Test-Path $python)) {
    $base = "$env:LOCALAPPDATA\Microsoft\WindowsApps\python.exe"
    if (-not (Test-Path $base)) { $base = "python.exe" }
    & $base -m venv "$repo\.venv-build"
}
# nvidia wheels = CUDA runtime DLLs bundled into the exe (GPU support)
& $python -m pip install --quiet -r requirements.txt pyinstaller nvidia-cublas-cu12 nvidia-cudnn-cu12

if ($LocalPack) {
    Write-Output "== Building local-inference pack zip =="
    $packVer = (Select-String -Path whisperflow\localpack.py -Pattern 'PACK_VERSION = "([^"]+)"').Matches[0].Groups[1].Value
    $packStage = "$repo\build\local-pack-stage"
    Remove-Item -Recurse -Force $packStage -ErrorAction SilentlyContinue
    New-Item -ItemType Directory -Force $packStage | Out-Null

    # Copy the SAME packages the "full" frozen build bundles, from the SAME
    # venv — this is what makes the native .pyd ABI-compatible with a
    # WF_BUILD=cloud exe built from this same venv.
    $sitePkgs = "$repo\.venv-build\Lib\site-packages"
    foreach ($pkg in @("faster_whisper", "ctranslate2", "tokenizers", "nvidia", "av", "onnxruntime")) {
        $src = "$sitePkgs\$pkg"
        if (Test-Path $src) { Copy-Item -Recurse $src "$packStage\$pkg" }
    }

    # Validate that all required packages were staged successfully
    $expectedPkgs = @("faster_whisper", "ctranslate2", "tokenizers", "nvidia", "av", "onnxruntime")
    $missing = @()
    foreach ($pkg in $expectedPkgs) {
        $stagedPath = "$packStage\$pkg"
        if (-not (Test-Path $stagedPath)) {
            $missing += $pkg
        }
    }
    if ($missing.Count -gt 0) {
        Write-Error "Local-pack staging failed: missing package(s): $($missing -join ', ')"
    }

    $zipPath = "$repo\dist\whisperflow-local-pack-v$packVer.zip"
    Remove-Item -Force $zipPath -ErrorAction SilentlyContinue
    Compress-Archive -Path "$packStage\*" -DestinationPath $zipPath
    $sha = (Get-FileHash $zipPath -Algorithm SHA256).Hash.ToLower()
    Set-Content -Path "$zipPath.sha256" -Value $sha -NoNewline -Encoding ascii

    $mb = [math]::Round((Get-Item $zipPath).Length / 1MB)
    Write-Output ""
    Write-Output "DONE: $zipPath ($mb MB), sha256=$sha"
    Write-Output "Publish both files as release assets: gh release upload vX.Y.Z `"$zipPath`" `"$zipPath.sha256`""
    exit 0
}

$env:WF_BUILD = if ($Full) { "full" } else { "cloud" }
Write-Output "== 1/2 PyInstaller freeze (WF_BUILD=$env:WF_BUILD) =="
& $python -m PyInstaller installer\whisperflow.spec --noconfirm --distpath dist --workpath build
$exeName = "WhisperFlow"
if (-not (Test-Path "$repo\dist\$exeName\$exeName.exe")) {
    Write-Error "PyInstaller did not produce dist\$exeName\$exeName.exe"
}

Write-Output "== 2/2 Inno Setup compile =="
$isccCandidates = @(
    "${env:ProgramFiles(x86)}\Inno Setup 6\ISCC.exe",
    "$env:ProgramFiles\Inno Setup 6\ISCC.exe",
    "$env:LOCALAPPDATA\Programs\Inno Setup 6\ISCC.exe"  # winget --scope user
)
$iscc = $isccCandidates | Where-Object { Test-Path $_ } | Select-Object -First 1
if (-not $iscc) {
    Write-Error "Inno Setup 6 not found. Install it: winget install JRSoftware.InnoSetup --scope user"
}
& $iscc "installer\whisperflow.iss"

$setup = "$repo\installer\Output\WhisperFlow-Setup.exe"
if (Test-Path $setup) {
    $mb = [math]::Round((Get-Item $setup).Length / 1MB)
    Write-Output ""
    Write-Output "DONE: $setup ($mb MB)"
    Write-Output "Distribute via: gh release create vX.Y.Z `"$setup`""
} else {
    Write-Error "Inno Setup did not produce $setup"
}
