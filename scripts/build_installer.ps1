# Builds WhisperFlow-Setup.exe end to end:
#   1. PyInstaller freeze  -> dist\WhisperFlow\WhisperFlow.exe
#   2. Inno Setup compile  -> installer\Output\WhisperFlow-Setup.exe
#
# One-time prerequisite on the build machine:
#   Inno Setup 6 — https://jrsoftware.org/isinfo.php
# The script creates/updates its own clean build venv (.venv-build) so the
# frozen exe contains ONLY the app's real dependencies — building from a
# global Python that has other packages installed produces a multi-GB dist.
#
# The single distributed build is cloud-only (~29MB installer): Groq, Gemini,
# OpenAI, Deepgram, NVIDIA. Local (on-device) inference works only when
# running from source with faster-whisper installed.
#
# Usage:
#   powershell -ExecutionPolicy Bypass -File scripts\build_installer.ps1

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
& $python -m pip install --quiet -r requirements.txt pyinstaller

Write-Output "== 1/2 PyInstaller freeze =="
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
