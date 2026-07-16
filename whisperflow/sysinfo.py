"""System probe + model recommendation.

Detects the machine's real capabilities (NVIDIA GPU + VRAM via nvidia-smi,
RAM, CPU cores) and recommends the best STT setup — so a user never has to
guess which model their laptop can actually run. Used by `app.py --recommend`
and by the startup sanity check (warns when config doesn't match hardware,
e.g. device="cuda" on a machine with no NVIDIA GPU).
"""

from __future__ import annotations

import ctypes
import logging
import os
import subprocess
import sys
import winreg
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)

# ---- autostart (per-user, windowless, reversible, no admin) ----------------
# HKCU Run key: Windows launches this command at login. app.py resolves
# config/history/logs from its own __file__, so the login-launched process
# (cwd = system32) still finds everything.
#
# Store Python gotcha: its pythonw.exe is a 0-byte app-execution alias that
# fails SILENTLY when launched from the Run key — the app simply never starts.
# For Store Python we therefore register wscript.exe + run.vbs (a real system
# exe launching the python.exe alias with a hidden window) instead.
_RUN_KEY = r"Software\Microsoft\Windows\CurrentVersion\Run"
_RUN_VALUE = "WhisperFlow"
_APP_ROOT = Path(__file__).resolve().parent.parent


def _pythonw_path() -> str:
    """pythonw.exe (windowless) matching the current interpreter."""
    exe = Path(sys.executable)
    if exe.name.lower() == "pythonw.exe":
        return str(exe)
    cand = exe.with_name("pythonw.exe")
    if cand.exists():
        return str(cand)
    fallback = Path(os.path.expandvars(r"%LOCALAPPDATA%\Microsoft\WindowsApps\pythonw.exe"))
    return str(fallback if fallback.exists() else cand)


def _is_store_python() -> bool:
    return "windowsapps" in str(Path(sys.executable)).lower()


def autostart_command() -> str:
    """Exact command written to the Run key."""
    if getattr(sys, "frozen", False):
        # installed (PyInstaller) build: the exe IS the app — no interpreter,
        # no run.vbs indirection needed
        return f'"{Path(sys.executable).resolve()}" --autostart'
    app_py = _APP_ROOT / "app.py"
    if _is_store_python():
        wscript = Path(os.path.expandvars(r"%SystemRoot%\System32\wscript.exe"))
        run_vbs = _APP_ROOT / "run.vbs"
        if wscript.exists() and run_vbs.exists():
            return f'"{wscript}" //B "{run_vbs}"'
        # last resort: the python.exe alias demonstrably launches (unlike
        # pythonw) but keeps a console window open for the app's lifetime
        log.warning("wscript.exe or run.vbs missing — autostart will show a console window")
        python = os.path.expandvars(r"%LOCALAPPDATA%\Microsoft\WindowsApps\python.exe")
        return f'"{python}" "{app_py}" --autostart'
    return f'"{_pythonw_path()}" "{app_py}" --autostart'


def get_autostart_command() -> str | None:
    """The raw command currently registered in the Run key, or None."""
    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, _RUN_KEY) as key:
            value, _ = winreg.QueryValueEx(key, _RUN_VALUE)
            return str(value)
    except FileNotFoundError:
        return None


def is_autostart_enabled() -> bool:
    return bool(get_autostart_command())


def ensure_autostart(sentinel: Path) -> None:
    """First-run registration + self-healing of stale entries.

    - no entry, no sentinel  -> true first run: register + write sentinel
    - no entry, sentinel     -> user opted out (tray toggle): leave it alone
    - entry != desired       -> stale (broken pythonw alias, moved repo,
                                changed interpreter): re-register
    """
    try:
        current = get_autostart_command()
        desired = autostart_command()
        if current is None:
            if sentinel.exists():
                return
            enable_autostart()
        elif current != desired:
            enable_autostart()
            log.info("autostart entry was stale — re-registered")
        if not sentinel.exists():
            sentinel.write_text("1", encoding="utf-8")
    except OSError as exc:  # registry/filesystem unavailable — non-fatal
        log.warning("could not ensure autostart: %s", exc)


def enable_autostart() -> None:
    with winreg.CreateKey(winreg.HKEY_CURRENT_USER, _RUN_KEY) as key:
        winreg.SetValueEx(key, _RUN_VALUE, 0, winreg.REG_SZ, autostart_command())
    log.info("autostart enabled: %s", autostart_command())


def disable_autostart() -> None:
    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, _RUN_KEY, 0, winreg.KEY_SET_VALUE) as key:
            winreg.DeleteValue(key, _RUN_VALUE)
        log.info("autostart disabled")
    except FileNotFoundError:
        pass


# ---- single-instance activation ---------------------------------------------
# A second `python app.py` launch can't start (named mutex) — instead it signals
# this named event and the running instance shows its main window. Standard
# "clicking the app again opens its window" behavior.
_SHOW_EVENT = "Global\\WhisperFlowShowMainWindow"
_WAIT_OBJECT_0 = 0
_EVENT_MODIFY_STATE = 0x0002


def create_show_event() -> int:
    """Create the named auto-reset event the running instance waits on."""
    return ctypes.windll.kernel32.CreateEventW(None, False, False, _SHOW_EVENT)


def signal_show_event() -> bool:
    """Ask the running instance to show its main window. False if none is running."""
    kernel32 = ctypes.windll.kernel32
    handle = kernel32.OpenEventW(_EVENT_MODIFY_STATE, False, _SHOW_EVENT)
    if not handle:
        return False
    try:
        return bool(kernel32.SetEvent(handle))
    finally:
        kernel32.CloseHandle(handle)


def wait_show_event(handle: int, timeout_ms: int) -> bool:
    return ctypes.windll.kernel32.WaitForSingleObject(handle, timeout_ms) == _WAIT_OBJECT_0


@dataclass
class SystemSpecs:
    gpu_name: str | None
    vram_mb: int  # 0 when no NVIDIA GPU
    ram_gb: float
    cpu_cores: int


@dataclass
class Recommendation:
    engine: str  # local | any registered provider id (see whisperflow.stt.providers)
    name: str  # registry model name (local) or cloud model id
    device: str  # cuda | cpu
    compute_type: str
    reason: str
    alternatives: list[str]


def probe() -> SystemSpecs:
    gpu_name, vram_mb = _probe_nvidia()
    return SystemSpecs(
        gpu_name=gpu_name,
        vram_mb=vram_mb,
        ram_gb=_probe_ram_gb(),
        cpu_cores=os.cpu_count() or 1,
    )


def _probe_nvidia() -> tuple[str | None, int]:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=10,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        if out.returncode != 0 or not out.stdout.strip():
            return None, 0
        name, mem = out.stdout.strip().splitlines()[0].rsplit(",", 1)
        return name.strip(), int(mem.strip())
    except (FileNotFoundError, subprocess.TimeoutExpired, ValueError):
        return None, 0


def _probe_ram_gb() -> float:
    class MEMORYSTATUSEX(ctypes.Structure):
        _fields_ = [
            ("dwLength", ctypes.c_ulong),
            ("dwMemoryLoad", ctypes.c_ulong),
            ("ullTotalPhys", ctypes.c_ulonglong),
            ("ullAvailPhys", ctypes.c_ulonglong),
            ("ullTotalPageFile", ctypes.c_ulonglong),
            ("ullAvailPageFile", ctypes.c_ulonglong),
            ("ullTotalVirtual", ctypes.c_ulonglong),
            ("ullAvailVirtual", ctypes.c_ulonglong),
            ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
        ]

    stat = MEMORYSTATUSEX()
    stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
    return stat.ullTotalPhys / (1024**3)


def _detect_cloud_provider() -> str:
    """Which cloud provider's API key the user actually has set in the
    environment (checked in provider-registry order), so recommend() can
    suggest a provider the user can use right now without any extra signup
    step. Falls back to "groq" — the best free "sign up now" default — when
    nobody has any cloud key set yet. Mirrors app.py's
    _any_cloud_api_key_available() (which only answers "does ANY key
    exist"); this answers "WHICH one"."""
    from whisperflow.stt import providers

    for p in providers.cloud_providers():
        if p.api_key_env and os.environ.get(p.api_key_env):
            return p.id
    return "groq"


def recommend(
    specs: SystemSpecs, has_api_key: bool = False, local_available: bool = True
) -> Recommendation:
    """Best model for THIS machine. Ladder:

    - NVIDIA >=5GB VRAM  -> large-v3-turbo cuda int8_float16 (proven default)
    - NVIDIA 3-5GB       -> medium cuda int8_float16
    - NVIDIA <3GB        -> small cuda int8_float16
    - No NVIDIA, >=8GB RAM + >=4 cores -> small cpu int8 (usable, slower)
    - Weak machine       -> BYOK cloud (Groq is the primary free-cloud
                            recommendation) if a key is available, else
                            smallest local with an honest warning

    local_available=False (this install doesn't include local inference)
    short-circuits the whole hardware ladder — even a strong GPU can't run
    a model that isn't there, so recommend the free cloud engine instead.
    """
    if not local_available:
        from whisperflow.stt import providers

        engine = _detect_cloud_provider() if has_api_key else "groq"
        provider = providers.get(engine)
        return Recommendation(
            engine=engine,
            name=provider.default_model,
            device="cpu",
            compute_type="int8",
            reason="Groq is free and instant, no download needed",
            alternatives=[],
        )

    alts: list[str] = []

    if specs.vram_mb >= 5000:
        alts.append("large-v3 (best Hindi accuracy, ~5x slower) if you can wait")
        if has_api_key:
            alts.append("engine='groq' to save all VRAM for other GPU work")
        return Recommendation(
            engine="local",
            name="large-v3-turbo",
            device="cuda",
            compute_type="int8_float16",
            reason=f"{specs.gpu_name} with {specs.vram_mb / 1024:.0f}GB VRAM runs the "
            "flagship turbo model comfortably (~1.5GB VRAM, ~6x realtime)",
            alternatives=alts,
        )

    if specs.vram_mb >= 3000:
        alts.append("small for faster response at lower accuracy")
        return Recommendation(
            engine="local",
            name="medium",
            device="cuda",
            compute_type="int8_float16",
            reason=f"{specs.gpu_name} has {specs.vram_mb / 1024:.1f}GB VRAM — medium fits; "
            "large-v3-turbo may OOM alongside other GPU apps",
            alternatives=alts,
        )

    if specs.vram_mb > 0:
        return Recommendation(
            engine="local",
            name="small",
            device="cuda",
            compute_type="int8_float16",
            reason=f"{specs.gpu_name} has only {specs.vram_mb / 1024:.1f}GB VRAM — small is the safe fit",
            alternatives=["engine='groq' (free) for better accuracy than small"],
        )

    if specs.ram_gb >= 8 and specs.cpu_cores >= 4:
        alts = [
            "engine='groq' (free, 2000/day) — instant cloud transcription, no download",
            "medium on cpu if accuracy matters more than speed and you want to stay offline",
        ]
        return Recommendation(
            engine="local",
            name="small",
            device="cpu",
            compute_type="int8",
            reason=f"no NVIDIA GPU detected; {specs.cpu_cores} cores / {specs.ram_gb:.0f}GB RAM "
            "can run small on CPU (expect a few seconds per dictation)",
            alternatives=alts,
        )

    if has_api_key:
        from whisperflow.stt import providers

        engine = _detect_cloud_provider()
        provider = providers.get(engine)
        return Recommendation(
            engine=engine,
            name=provider.default_model,
            device="cpu",
            compute_type="int8",
            reason=f"this machine ({specs.cpu_cores} cores, {specs.ram_gb:.0f}GB RAM, no NVIDIA GPU) "
            f"is too weak for a good local model — free cloud ({provider.display_name}) is the honest "
            "recommendation (note: audio leaves the machine)",
            alternatives=["small on cpu (fully private but slow and less accurate)"],
        )

    return Recommendation(
        engine="local",
        name="small",
        device="cpu",
        compute_type="int8",
        reason=f"this machine ({specs.cpu_cores} cores, {specs.ram_gb:.0f}GB RAM, no NVIDIA GPU) "
        "will run small slowly — consider a free Groq key for instant cloud transcription instead",
        alternatives=["engine='groq' — free, 2000 requests/day, no local download needed"],
    )


def _config_and_providers():
    """Shared lazy import for build_recommended_config/build_config_for_engine
    — both need Config + the provider registry but import lazily (module
    scope would risk a circular import with whisperflow.config/stt)."""
    from whisperflow.config import Config
    from whisperflow.stt import providers

    return Config, providers


def build_recommended_config(rec: Recommendation):
    """Pure: build a Config from a Recommendation, no file I/O. Used by both
    the unattended `--headless` first-run path (app.py bootstrap_config) and
    the interactive first-run chooser's "Use recommended" button."""
    Config, providers = _config_and_providers()

    cfg = Config()
    cfg.model.engine = rec.engine
    if rec.engine == "local":
        cfg.model.name = rec.name
        cfg.model.device = rec.device
        cfg.model.compute_type = rec.compute_type
    else:
        cfg.model.cloud_model = rec.name
        cfg.model.api_key_env = providers.get(rec.engine).api_key_env
    return cfg


def build_config_for_engine(engine_id: str, specs: SystemSpecs):
    """Pure: build a Config for a user's MANUAL provider pick (first-run
    chooser or Settings), as opposed to the system's auto-recommendation.

    For "local" this reuses recommend()'s hardware-tiered sizing rather than
    duplicating the VRAM ladder: recommend(specs, has_api_key=False) always
    returns engine="local" (the only non-local branch requires
    has_api_key=True), so its name/device/compute_type are exactly the
    right local sizing for this machine regardless of why the caller wants
    local.
    """
    if engine_id == "local":
        return build_recommended_config(recommend(specs, has_api_key=False))
    Config, providers = _config_and_providers()

    provider = providers.get(engine_id)
    cfg = Config()
    cfg.model.engine = engine_id
    cfg.model.cloud_model = provider.default_model
    cfg.model.api_key_env = provider.api_key_env
    return cfg


def startup_check(cfg_model, specs: SystemSpecs) -> str | None:
    """One-line warning when config mismatches hardware, else None."""
    if cfg_model.engine == "local" and cfg_model.device == "cuda" and specs.vram_mb == 0:
        return (
            "config sets device='cuda' but no NVIDIA GPU was detected — "
            "run 'python app.py --recommend' for a model suited to this machine"
        )
    if cfg_model.engine == "local" and cfg_model.name in ("large-v3-turbo", "large-v3") and 0 < specs.vram_mb < 3000:
        return (
            f"config model '{cfg_model.name}' needs ~1.5-3GB VRAM but only "
            f"{specs.vram_mb / 1024:.1f}GB detected — run 'python app.py --recommend'"
        )
    return None
