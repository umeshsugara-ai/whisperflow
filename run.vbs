' WhisperFlow launcher — starts the dictation daemon without a console window.
' To autostart on boot: Win+R -> shell:startup -> place a shortcut to this file.
Dim shell, appDir
Set shell = CreateObject("WScript.Shell")
appDir = Left(WScript.ScriptFullName, InStrRev(WScript.ScriptFullName, "\") - 1)
shell.CurrentDirectory = appDir
shell.Run """C:\Users\Lenovo\AppData\Local\Microsoft\WindowsApps\pythonw.exe"" """ & appDir & "\app.py""", 0, False
