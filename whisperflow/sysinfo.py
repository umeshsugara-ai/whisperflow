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
# HKCU Run key: Windows launches this command at login. pythonw.exe gives no
# console window; app.py resolves config/history/logs from its own __file__, so
# the login-launched process (cwd = system32) still finds everything.
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


def autostart_command() -> str:
    """Exact command written to the Run key — quoted pythonw + quoted app.py."""
    return f'"{_pythonw_path()}" "{_APP_ROOT / "app.py"}"'


def is_autostart_enabled() -> bool:
    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, _RUN_KEY) as key:
            value, _ = winreg.QueryValueEx(key, _RUN_VALUE)
            return bool(value)
    except FileNotFoundError:
        return False


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


@dataclass
class SystemSpecs:
    gpu_name: str | None
    vram_mb: int  # 0 when no NVIDIA GPU
    ram_gb: float
    cpu_cores: int


@dataclass
class Recommendation:
    engine: str  # local | gemini
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


def recommend(specs: SystemSpecs, has_api_key: bool = False) -> Recommendation:
    """Best model for THIS machine. Ladder:

    - NVIDIA >=5GB VRAM  -> large-v3-turbo cuda int8_float16 (proven default)
    - NVIDIA 3-5GB       -> medium cuda int8_float16
    - NVIDIA <3GB        -> small cuda int8_float16
    - No NVIDIA, >=8GB RAM + >=4 cores -> small cpu int8 (usable, slower)
    - Weak machine       -> BYOK cloud (gemini) if key available, else
                            smallest local with an honest warning
    """
    alts: list[str] = []

    if specs.vram_mb >= 5000:
        alts.append("large-v3 (best Hindi accuracy, ~5x slower) if you can wait")
        if has_api_key:
            alts.append("engine='gemini' to save all VRAM for other GPU work")
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
            alternatives=["engine='gemini' (BYOK) for better accuracy than small"] if has_api_key else [],
        )

    if specs.ram_gb >= 8 and specs.cpu_cores >= 4:
        alts = ["medium on cpu if accuracy matters more than speed"]
        if has_api_key:
            alts.insert(0, "engine='gemini' (BYOK) — much better accuracy than small, no download")
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
        return Recommendation(
            engine="gemini",
            name="gemini-2.5-flash",
            device="cpu",
            compute_type="int8",
            reason=f"this machine ({specs.cpu_cores} cores, {specs.ram_gb:.0f}GB RAM, no NVIDIA GPU) "
            "is too weak for a good local model — BYOK cloud is the honest recommendation "
            "(note: audio leaves the machine)",
            alternatives=["small on cpu (fully private but slow and less accurate)"],
        )

    return Recommendation(
        engine="local",
        name="small",
        device="cpu",
        compute_type="int8",
        reason=f"this machine ({specs.cpu_cores} cores, {specs.ram_gb:.0f}GB RAM, no NVIDIA GPU) "
        "will be slow — consider [model].engine='gemini' with your own API key for better quality",
        alternatives=["engine='gemini' (set GEMINI_API_KEY) — better accuracy, but audio goes to Google"],
    )


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
