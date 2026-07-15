' WhisperFlow product-window launcher — for the Start Menu shortcut / taskbar
' pin, so clicking the icon behaves like Wispr Flow's: opens the main window
' if not running, or brings the main window of an already-running instance
' to the front (see app.py's Global\WhisperFlowShowMainWindow event).
' Unlike run.vbs (used by the Windows-logon autostart entry), this does NOT
' pass --autostart, so app.py opens the main window instead of staying
' pill-only.
Dim shell, fso, appDir, python
Set shell = CreateObject("WScript.Shell")
Set fso = CreateObject("Scripting.FileSystemObject")
appDir = Left(WScript.ScriptFullName, InStrRev(WScript.ScriptFullName, "\") - 1)
shell.CurrentDirectory = appDir
python = shell.ExpandEnvironmentStrings("%LOCALAPPDATA%") & "\Microsoft\WindowsApps\python.exe"
If Not fso.FileExists(python) Then python = "python.exe"
shell.Run """" & python & """ """ & appDir & "\app.py""", 0, False
