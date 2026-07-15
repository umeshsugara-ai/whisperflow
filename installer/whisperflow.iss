; WhisperFlow — Inno Setup 6 script (industry-style setup wizard).
; Compile: ISCC.exe installer\whisperflow.iss   (or scripts\build_installer.ps1)
; Input : dist\WhisperFlow\  (PyInstaller onedir output — build that first)
; Output: installer\Output\WhisperFlow-Setup.exe
;
; Per-user install (no UAC prompt), like Wispr Flow itself: the app lands in
; %LOCALAPPDATA%\Programs\WhisperFlow, writable state in %LOCALAPPDATA%\WhisperFlow.

#define AppName "WhisperFlow"
#define AppVersion "1.0.0"
#define AppExe "WhisperFlow.exe"
#define AppPublisher "Vidysea"

[Setup]
AppId={{6E1B62F3-7C9A-4D2B-9B1E-A3F41C0D8E52}}
AppName={#AppName}
AppVersion={#AppVersion}
AppPublisher={#AppPublisher}
DefaultDirName={autopf}\{#AppName}
DefaultGroupName={#AppName}
PrivilegesRequired=lowest
DisableProgramGroupPage=yes
OutputDir=Output
OutputBaseFilename=WhisperFlow-Setup
SetupIconFile=..\assets\app.ico
UninstallDisplayIcon={app}\{#AppExe}
Compression=lzma2
SolidCompression=yes
WizardStyle=modern
; the model downloads on first run, so the installer itself stays small-ish
ArchitecturesInstallIn64BitMode=x64compatible

[Tasks]
Name: "autostart"; Description: "Start {#AppName} automatically when Windows starts (recommended)"
Name: "desktopicon"; Description: "Create a &desktop shortcut"; Flags: unchecked

[Files]
Source: "..\dist\WhisperFlow\*"; DestDir: "{app}"; Flags: recursesubdirs ignoreversion

[Icons]
Name: "{userprograms}\{#AppName}"; Filename: "{app}\{#AppExe}"
Name: "{userdesktop}\{#AppName}"; Filename: "{app}\{#AppExe}"; Tasks: desktopicon

[Registry]
Root: HKCU; Subkey: "Software\Microsoft\Windows\CurrentVersion\Run"; \
    ValueType: string; ValueName: "{#AppName}"; \
    ValueData: """{app}\{#AppExe}"" --autostart"; \
    Tasks: autostart; Flags: uninsdeletevalue

[Run]
Filename: "{app}\{#AppExe}"; Description: "Launch {#AppName} now (first run downloads the ~1.5GB speech model)"; \
    Flags: nowait postinstall skipifsilent

[Code]
// On uninstall, offer the industry-standard "keep my data?" choice for
// %LOCALAPPDATA%\WhisperFlow (config, dictation history, logs).
procedure CurUninstallStepChanged(CurUninstallStep: TUninstallStep);
var
  DataDir: string;
begin
  if CurUninstallStep = usPostUninstall then
  begin
    DataDir := ExpandConstant('{localappdata}\WhisperFlow');
    // silent uninstall must never destroy data — only ask interactively
    if DirExists(DataDir) and not UninstallSilent then
    begin
      if MsgBox('Also delete your WhisperFlow data (settings, dictation history)?'
                + #13#10 + DataDir,
                mbConfirmation, MB_YESNO) = IDYES then
        DelTree(DataDir, True, True, True);
    end;
  end;
end;
