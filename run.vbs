' WhisperFlow launcher — starts the dictation daemon without a console window.
' Registered in the HKCU Run key by the app itself (whisperflow/sysinfo.py):
' wscript.exe //B run.vbs. Launches the python.exe alias (Store Python's
' pythonw.exe alias fails silently at logon) with the window hidden.
' Also works from shell:startup or a double-click.
Dim shell, fso, appDir, python
Set shell = CreateObject("WScript.Shell")
Set fso = CreateObject("Scripting.FileSystemObject")
appDir = Left(WScript.ScriptFullName, InStrRev(WScript.ScriptFullName, "\") - 1)
shell.CurrentDirectory = appDir
python = shell.ExpandEnvironmentStrings("%LOCALAPPDATA%") & "\Microsoft\WindowsApps\python.exe"
If Not fso.FileExists(python) Then python = "python.exe"
shell.Run """" & python & """ """ & appDir & "\app.py"" --autostart", 0, False
