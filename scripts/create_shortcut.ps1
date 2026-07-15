# Creates a Start Menu shortcut that opens WhisperFlow's product window
# (main window / pill), not a raw console — safe to run on any machine, any
# clone path; it resolves everything relative to this script's location.
#
# Usage:
#   powershell -ExecutionPolicy Bypass -File scripts\create_shortcut.ps1
#   powershell -ExecutionPolicy Bypass -File scripts\create_shortcut.ps1 -Name "myWhisperFlow"
#
# After it runs: Start Menu -> find the shortcut -> right-click -> "Pin to taskbar".

param(
    [string]$Name = "WhisperFlow"
)

$repoRoot = Split-Path -Parent $PSScriptRoot
$icon = Join-Path $repoRoot "assets\app.ico"
$launcher = Join-Path $repoRoot "launch_ui.vbs"

if (-not (Test-Path $icon)) {
    Write-Output "assets\app.ico not found — generating it..."
    $python = "$env:LOCALAPPDATA\Microsoft\WindowsApps\python.exe"
    if (-not (Test-Path $python)) { $python = "python.exe" }
    $env:PYTHONPATH = $repoRoot
    & $python (Join-Path $repoRoot "scripts\make_icon.py")
}

$shortcutPath = "$env:APPDATA\Microsoft\Windows\Start Menu\Programs\$Name.lnk"
$sh = New-Object -ComObject WScript.Shell
$lnk = $sh.CreateShortcut($shortcutPath)
$lnk.TargetPath = "$env:SystemRoot\System32\wscript.exe"
$lnk.Arguments = "//B `"$launcher`""
$lnk.WorkingDirectory = $repoRoot
$lnk.IconLocation = $icon
$lnk.Description = "$Name — local dictation"
$lnk.Save()

Write-Output "Created: $shortcutPath"
Write-Output "Next: open Start Menu, find '$Name', right-click -> Pin to taskbar."
