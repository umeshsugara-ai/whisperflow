; WhisperFlow — Inno Setup 6 script (industry-style setup wizard).
; Compile: scripts\build_installer.ps1  (or ISCC.exe /DAppVersion=x.y.z installer\whisperflow.iss)
; Input : dist\WhisperFlow\  (PyInstaller onedir output — build that first)
; Output: installer\Output\WhisperFlow-Setup.exe
;
; Per-user install (no UAC prompt), like Wispr Flow itself: the app lands in
; %LOCALAPPDATA%\Programs\WhisperFlow, writable state in %LOCALAPPDATA%\WhisperFlow.

#define AppName "WhisperFlow"
#ifndef AppVersion
  ; The version lives ONLY in whisperflow/__init__.py; the build script passes
  ; it via /DAppVersion. A hand-compiled installer stamped with a stale or
  ; zero version would make every user's auto-updater misjudge the shipped
  ; release forever — fail the compile loudly instead of guessing.
  #error AppVersion not defined - build via scripts\build_installer.ps1 (or pass /DAppVersion=x.y.z)
#endif
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
; No skipifsilent: the auto-updater upgrades with /VERYSILENT and relies on
; this default-checked postinstall entry to relaunch the app afterwards.
; Interactive installs still get the same "Launch now" checkbox as before.
Filename: "{app}\{#AppExe}"; Description: "Launch {#AppName} now"; \
    Flags: nowait postinstall

[Code]
// Force-close any running WhisperFlow instance BEFORE uninstall touches
// files — otherwise Windows can't delete the locked exe/DLLs (deferred to
// next reboot) and the tray icon/pill keeps running, invisible-uninstall.
// The app has no unsaved in-memory state to lose (history/config are
// persisted continuously), so a hard kill is safe here.
function InitializeUninstall(): Boolean;
var
  ResultCode: Integer;
begin
  // watchdog FIRST — killed second, it would read the app's forced exit as
  // a crash and relaunch it in the middle of the uninstall
  Exec('taskkill.exe', '/F /IM WhisperFlowWatchdog.exe', '', SW_HIDE, ewWaitUntilTerminated, ResultCode);
  Exec('taskkill.exe', '/F /IM {#AppExe}', '', SW_HIDE, ewWaitUntilTerminated, ResultCode);
  Result := True;
end;

// Same for UPGRADE installs: running the new setup over an installed,
// running copy would hit locked exe/DLLs ("file in use" / files deferred
// to reboot). Kill it before files are copied; [Run] relaunches after.
function PrepareToInstall(var NeedsRestart: Boolean): String;
var
  ResultCode: Integer;
begin
  // watchdog first — see InitializeUninstall
  Exec('taskkill.exe', '/F /IM WhisperFlowWatchdog.exe', '', SW_HIDE, ewWaitUntilTerminated, ResultCode);
  Exec('taskkill.exe', '/F /IM {#AppExe}', '', SW_HIDE, ewWaitUntilTerminated, ResultCode);
  Result := '';
end;

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
