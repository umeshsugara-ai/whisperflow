"""WhisperFlow entry point.

Loads config -> loads the STT model once (held in VRAM) -> starts the hotkey
listener + controller worker. Run with --headless for the no-UI pipeline
(tray + overlay wiring arrives with the UI milestone).

    python app.py --headless
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
if sys.stdout:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from whisperflow.audio import Recorder
from whisperflow.config import DEFAULT_CONFIG_PATH, data_dir, load_config, load_dotenv
from whisperflow.controller import Controller, DictationResult, State
from whisperflow.dictionary import vocabulary_prompt
from whisperflow.history import History
from whisperflow.hotkey import HotkeyListener
from whisperflow.inject import injector
from whisperflow.processing import build_processor
from whisperflow.stt.registry import create_engine

log = logging.getLogger("whisperflow")

APP_ROOT = Path(__file__).resolve().parent


MUTEX_NAME = "Global\\WhisperFlowSingleInstance"
ERROR_ALREADY_EXISTS = 183


def acquire_single_instance() -> bool:
    """Named mutex — False if another WhisperFlow instance already runs."""
    import ctypes

    ctypes.windll.kernel32.CreateMutexW(None, False, MUTEX_NAME)
    return ctypes.windll.kernel32.GetLastError() != ERROR_ALREADY_EXISTS


# Last N WARNING+ records, surfaced in plain language on the main window's
# Home screen so non-technical users never have to open the raw log file.
_recent_warnings: "deque[str]" = None  # type: ignore[assignment]  # built in setup_logging


class _WarningBuffer(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        try:
            stamp = time.strftime("%d %b %H:%M", time.localtime(record.created))
            _recent_warnings.append(f"{stamp} — {record.getMessage()}")
        except Exception:  # noqa: BLE001 — logging must never crash the app
            pass


def recent_warnings() -> list[str]:
    return list(_recent_warnings) if _recent_warnings is not None else []


def setup_logging() -> None:
    from collections import deque
    from logging.handlers import RotatingFileHandler

    global _recent_warnings
    _recent_warnings = deque(maxlen=50)

    log_dir = data_dir() / "logs"
    log_dir.mkdir(exist_ok=True)
    warn_buffer = _WarningBuffer()
    warn_buffer.setLevel(logging.WARNING)
    handlers: list[logging.Handler] = [
        RotatingFileHandler(
            log_dir / "whisperflow.log", maxBytes=1_000_000, backupCount=3, encoding="utf-8"
        ),
        warn_buffer,
    ]
    if sys.stdout is not None:  # pythonw.exe has no console
        handlers.append(logging.StreamHandler())
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        handlers=handlers,
    )


def warmup(engine) -> None:
    """Transcribe 0.5s of silence so CUDA kernels are compiled before the
    first real dictation (saves ~0.5-1s on the first use)."""
    import numpy as np

    t0 = time.perf_counter()
    engine.transcribe(np.zeros(8000, dtype=np.float32))
    log.info("CUDA warmup done in %.1fs", time.perf_counter() - t0)


def _model_needs_download(model_cfg) -> bool:
    """True when the local faster-whisper model isn't in the HF cache yet."""
    if model_cfg.engine != "local":
        return False
    from whisperflow.stt.registry import resolve_model_id

    try:
        from huggingface_hub.constants import HF_HUB_CACHE

        cache = Path(HF_HUB_CACHE)
    except Exception:  # noqa: BLE001 — heuristic only, never block startup
        cache = Path.home() / ".cache" / "huggingface" / "hub"
    repo = resolve_model_id(model_cfg.name)
    return not (cache / ("models--" + repo.replace("/", "--"))).exists()


def _local_pack_needs_download(model_cfg) -> bool:
    """True when engine="local" but faster_whisper isn't importable AND the
    on-demand pack hasn't been downloaded yet (WF_BUILD=cloud installs only
    — a no-op check on a dev checkout or a WF_BUILD=full build)."""
    if model_cfg.engine != "local":
        return False
    from whisperflow.stt import registry

    try:
        registry._try_import_faster_whisper()
        return False
    except ImportError:
        pass
    from whisperflow import localpack

    return not localpack.is_installed()


def build_controller(cfg) -> tuple[Controller, HotkeyListener, History]:
    if _model_needs_download(cfg.model):
        # WARNING so it also lands on the main window's Home status strip
        log.warning(
            "Downloading the speech model %r (~1.5GB) — first run only, "
            "please keep the app open; dictation starts when it finishes.",
            cfg.model.name,
        )
    if _local_pack_needs_download(cfg.model):
        log.warning(
            "Local (on-device) mode isn't set up on this install (needs a one-time "
            "~800MB download) — startup will offer to switch engines."
        )
    engine = create_engine(cfg.model)
    engine.load()
    warmup(engine)

    recorder = Recorder(cfg.audio)
    history = History(data_dir() / "history.jsonl", max_entries=cfg.history.max_entries)

    def on_result(result: DictationResult) -> None:
        history.append(
            raw=result.raw_text,
            injected=result.injected_text,
            tier=result.cleanup_tier,
            method=result.method,
            language=result.language,
            duration_s=result.duration_s,
            latency_ms=result.transcribe_seconds * 1000.0,
        )

    from whisperflow.inject import focus

    target = {"hwnd": 0}  # last real (non-own) foreground window seen while recording

    def remember_target() -> None:
        hwnd = focus.current_window()
        if not focus.is_own_window(hwnd):
            target["hwnd"] = hwnd

    ctl = Controller(
        recorder=recorder,
        engine=engine,
        inject_text=lambda text: focus.inject_guarded(text, cfg.inject, target["hwnd"]),
        process_text=build_processor(
            cfg.cleanup, cfg.dictionary,
            gemini_api_key=cfg.model.resolve_api_key(), gemini_model=cfg.cleanup.gemini_model,
        ),
        on_result=on_result,
        language=cfg.model.language,
        initial_prompt=vocabulary_prompt(cfg.dictionary),
    )
    ctl.remember_target = remember_target  # called by state handlers on RECORDING

    def _track_target_while_recording() -> None:
        # keep re-capturing the target for as long as recording is active, so
        # clicking into the real destination window AFTER starting dictation
        # (e.g. via the pill) still lands the text there, instead of freezing
        # on whatever was focused the instant recording began.
        while True:
            time.sleep(0.25)
            if ctl.state is State.RECORDING:
                remember_target()

    threading.Thread(target=_track_target_while_recording, daemon=True, name="wf-target-tracker").start()

    listener = HotkeyListener(
        combo=cfg.hotkey.combo,
        tap_threshold_ms=cfg.hotkey.tap_threshold_ms,
        on_event=ctl.handle_hotkey,
        double_tap_ms=cfg.hotkey.double_tap_ms,
    )
    return ctl, listener, history


def run_headless(cfg, ctl, listener) -> int:
    def print_state(state: State, detail: str) -> None:
        if state is State.RECORDING:
            ctl.remember_target()
        log.info("state: %s %s", state.name, f"({detail})" if detail else "")

    ctl.on_state = print_state
    ctl.start()
    listener.start()

    log.info("ready — hotkey %s (tap=toggle, hold=push-to-talk, Esc=cancel). Ctrl+C to exit.", cfg.hotkey.combo)
    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        listener.stop()
        ctl.shutdown()
    return 0


def run_with_ui(cfg, ctl, listener, history, autostarted: bool = False, root=None) -> int:
    import threading
    import tkinter as tk

    from whisperflow import sysinfo
    from whisperflow.processing import build_processor
    from whisperflow.ui.overlay import Overlay
    from whisperflow.ui.tray import Tray

    if root is None:
        root = tk.Tk()
        root.withdraw()  # the root stays hidden — MainWindow/overlay are Toplevels

    def _log_tk_exception(exc_type, exc_value, exc_tb) -> None:
        # default handler prints to the (hidden) console — invisible when
        # launched via run.vbs; route UI callback crashes into the log instead
        log.error("UI callback failed", exc_info=(exc_type, exc_value, exc_tb))

    root.report_callback_exception = _log_tk_exception
    from whisperflow.hotkey import HotkeyEvent, format_hotkey_label

    overlay = Overlay(root)
    overlay.hotkey_label = format_hotkey_label(cfg.hotkey.combo)  # pill shows the real combo
    overlay.persistent = cfg.overlay.always_visible
    overlay.level_source = lambda: ctl.recorder.last_peak  # live waveform
    overlay.on_cancel = lambda: ctl.handle_hotkey(HotkeyEvent.RECORD_CANCEL)
    overlay.on_confirm = lambda: ctl.handle_hotkey(HotkeyEvent.RECORD_STOP)
    overlay.on_start = lambda: ctl.handle_hotkey(HotkeyEvent.RECORD_START)  # click pill to start

    def on_state(state: State, detail: str) -> None:
        tray.set_state(state.name, detail)
        if state is State.RECORDING:
            ctl.remember_target()
            overlay.post(overlay.show_recording, detail)
        elif state in (State.TRANSCRIBING, State.INJECTING):
            overlay.post(overlay.show_processing)
        elif state is State.ERROR:
            overlay.post(overlay.flash_error, f"Error: {detail}")
        else:  # IDLE
            if "clipboard" in detail:
                overlay.post(overlay.flash_warn, "Copied — press Ctrl+V")
            elif detail.startswith("injected"):
                overlay.post(overlay.flash_done, "Injected ✓")
            else:
                overlay.post(overlay.show_idle)

    def rebuild_processor() -> None:
        ctl.process_text = build_processor(
            cfg.cleanup, cfg.dictionary,
            gemini_api_key=cfg.model.resolve_api_key(), gemini_model=cfg.cleanup.gemini_model,
        )

    def on_tier_change(tier: str) -> None:
        rebuild_processor()

    def apply_live() -> None:
        """Apply the live-appliable parts of cfg (after a Settings save or a
        file reload). Model/hotkey changes still need a restart."""
        rebuild_processor()
        ctl.initial_prompt = vocabulary_prompt(cfg.dictionary)
        overlay.persistent = cfg.overlay.always_visible
        overlay.hotkey_label = format_hotkey_label(cfg.hotkey.combo)
        if ctl.state is State.IDLE:  # don't disturb an active recording UI
            root.after(0, overlay.show_idle)

    def on_reload_config() -> None:
        try:
            fresh = load_config(cfg.path)
        except Exception as exc:  # noqa: BLE001
            log.error("config reload failed: %s", exc)
            overlay.post(overlay.flash_error, f"Config error: {exc}")
            return
        # live-applicable settings (model/device changes need a restart)
        cfg.cleanup = fresh.cleanup
        cfg.inject = fresh.inject
        cfg.dictionary = fresh.dictionary
        cfg.audio = fresh.audio
        cfg.overlay = fresh.overlay
        apply_live()
        log.info("config reloaded (model/hotkey changes need restart)")

    def on_quit() -> None:
        root.after(0, root.quit)

    def on_open_main(tab: str = "home") -> None:
        from whisperflow.ui.main_window import MainWindow

        log.info("opening main window (tab=%s)", tab)
        root.after(
            0,
            lambda: MainWindow.open(
                root, cfg, history,
                apply_config=apply_live,
                warnings_source=recent_warnings,
                tab=tab,
            ),
        )

    def _watch_show_requests() -> None:
        """A second `python app.py` launch signals this event — show the window."""
        handle = sysinfo.create_show_event()
        if not handle:
            return
        while True:
            if sysinfo.wait_show_event(handle, 2000):
                on_open_main("home")

    overlay.on_open_main = lambda: on_open_main("home")  # right-click the pill

    tray = Tray(cfg, history, on_reload_config, on_quit, on_tier_change, on_open_main)
    ctl.on_state = on_state
    ctl.start()
    listener.start()
    tray.run_detached()
    threading.Thread(target=_watch_show_requests, daemon=True, name="wf-show-event").start()
    if cfg.overlay.always_visible:
        root.after(0, lambda: overlay.show_idle(hint=cfg.overlay.show_hint))
    if not autostarted:
        # a deliberate launch means the user wants to see the product screen;
        # the quiet path is logon autostart (pill only)
        on_open_main("home")

    log.info("ready — hotkey %s (tap=toggle, hold=push-to-talk, Esc=cancel)", cfg.hotkey.combo)
    try:
        root.mainloop()
    finally:
        listener.stop()
        ctl.shutdown()
        tray.stop()
    return 0


def _any_cloud_api_key_available() -> bool:
    """True when the user has an API key env var set for ANY registered cloud
    provider (not just Gemini) — used to decide whether cloud STT can be
    recommended as the default on a weak/GPU-less machine."""
    from whisperflow.stt import providers

    return any(os.environ.get(p.api_key_env) for p in providers.cloud_providers() if p.api_key_env)


def bootstrap_config(path: Path):
    """First run with no config.toml (installed build, or --headless):
    probe the hardware, generate a config from the recommendation, and save
    it — used only when there's no interactive first-run chooser (headless
    mode) or as the chooser's own "Use recommended" action."""
    from whisperflow import sysinfo
    from whisperflow.config import save_config

    specs = sysinfo.probe()
    rec = sysinfo.recommend(specs, has_api_key=_any_cloud_api_key_available())
    cfg = sysinfo.build_recommended_config(rec)
    cfg.path = path
    save_config(cfg, path)
    log.info(
        "first run — generated %s for %s (%s)",
        path.name,
        specs.gpu_name or f"CPU ({specs.cpu_cores} cores, {specs.ram_gb:.0f}GB RAM)",
        rec.reason,
    )
    return cfg


def print_recommendation() -> int:
    from whisperflow import sysinfo

    specs = sysinfo.probe()
    has_key = _any_cloud_api_key_available()
    rec = sysinfo.recommend(specs, has_api_key=has_key)

    print("System detected:")
    print(f"  GPU : {specs.gpu_name or 'none (no NVIDIA GPU found)'}"
          + (f" — {specs.vram_mb / 1024:.1f}GB VRAM" if specs.vram_mb else ""))
    print(f"  RAM : {specs.ram_gb:.0f}GB   CPU cores: {specs.cpu_cores}")
    print()
    print("Recommended config.toml [model] settings:")
    print(f"  engine = \"{rec.engine}\"")
    if rec.engine == "local":
        print(f"  name = \"{rec.name}\"")
        print(f"  device = \"{rec.device}\"")
        print(f"  compute_type = \"{rec.compute_type}\"")
    else:
        print(f"  cloud_model = \"{rec.name}\"")
        print("  # set your key: [model].api_key or the GEMINI_API_KEY env var")
        print("  # NOTE: cloud engine sends dictation audio to Google")
    print(f"\nWhy: {rec.reason}")
    if rec.alternatives:
        print("\nAlternatives:")
        for alt in rec.alternatives:
            print(f"  - {alt}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--headless", action="store_true", help="run without tray/overlay UI")
    ap.add_argument("--recommend", action="store_true", help="detect hardware and suggest the best model, then exit")
    ap.add_argument("--install-autostart", action="store_true", help="register WhisperFlow to start at Windows login, then exit")
    ap.add_argument("--uninstall-autostart", action="store_true", help="remove the Windows login autostart entry, then exit")
    ap.add_argument("--autostart", action="store_true", help=argparse.SUPPRESS)  # set by the logon Run entry
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    load_dotenv()  # .env next to app.py — the easy home for GEMINI_API_KEY

    if args.recommend:
        if sys.stdout:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        return print_recommendation()

    if args.install_autostart or args.uninstall_autostart:
        from whisperflow import sysinfo

        if args.install_autostart:
            sysinfo.enable_autostart()
            print(f"Autostart enabled — WhisperFlow will start at login:\n  {sysinfo.autostart_command()}")
        else:
            sysinfo.disable_autostart()
            print("Autostart disabled — WhisperFlow will no longer start at login.")
        return 0

    setup_logging()

    if args.autostart:
        log.info("started via Windows logon autostart")

    if not acquire_single_instance():
        from whisperflow import sysinfo

        if sysinfo.signal_show_event():
            log.info("already running — asked the running instance to show its window")
            print("WhisperFlow is already running — opening its window.")
            return 0
        log.error("WhisperFlow is already running — exiting.")
        print("WhisperFlow is already running.")
        return 2

    cfg_path = Path(args.config) if args.config else DEFAULT_CONFIG_PATH
    first_run_root = None
    if not cfg_path.exists():
        if args.headless:
            cfg = bootstrap_config(cfg_path)  # unattended — no display to show a chooser on
        else:
            import tkinter as tk

            from whisperflow import sysinfo
            from whisperflow.ui.first_run import show_first_run_chooser

            first_run_root = tk.Tk()
            first_run_root.withdraw()
            specs = sysinfo.probe()
            rec = sysinfo.recommend(specs, has_api_key=_any_cloud_api_key_available())
            cfg = show_first_run_chooser(first_run_root, specs, rec, cfg_path)
            log.info("first run — user chose %s via the chooser dialog", cfg.model.engine)
    else:
        cfg = load_config(cfg_path)
    log.info(
        "config: engine=%s model=%s hotkey=%s cleanup=%s",
        cfg.model.engine,
        cfg.model.cloud_model if cfg.model.engine != "local" else cfg.model.name,
        cfg.hotkey.combo,
        cfg.cleanup.tier,
    )

    from whisperflow import sysinfo

    # Autostart: register on first run and self-heal stale entries (e.g. the
    # broken Store-Python pythonw command) so WhisperFlow reappears after a
    # reboot (Wispr-style). Sentinel-gated so a later opt-out via the tray is
    # never overridden. Disable entirely with [startup].auto_register = false.
    if cfg.startup.auto_register:
        sysinfo.ensure_autostart(data_dir() / ".autostart_initialized")

    warning = sysinfo.startup_check(cfg.model, sysinfo.probe())
    if warning:
        log.warning(warning)

    try:
        ctl, listener, history = build_controller(cfg)
    except RuntimeError as exc:
        log.error("startup failed: %s", exc)
        if args.headless:
            raise
        import tkinter as tk

        root = first_run_root if first_run_root is not None else tk.Tk()
        if first_run_root is None:
            root.withdraw()

        # A saved engine that isn't available on this build (e.g. "local" on
        # a cloud-only install with no pack yet) must never be a dead end —
        # let the user pick another engine right now instead of just dying.
        # Reuses the first-run chooser as a recovery UI; it saves whatever
        # the user picks, so this also satisfies "a setup-time choice always
        # updates the saved config" (existing unrelated settings untouched).
        from whisperflow.ui.first_run import show_first_run_chooser

        log.warning(
            "saved engine %r isn't available on this build — opening the engine "
            "picker so you can choose another",
            cfg.model.engine,
        )
        specs = sysinfo.probe()
        rec = sysinfo.recommend(specs, has_api_key=_any_cloud_api_key_available())
        cfg = show_first_run_chooser(root, specs, rec, cfg_path)
        try:
            ctl, listener, history = build_controller(cfg)
        except RuntimeError as exc2:
            log.error("startup still failed after re-picking engine: %s", exc2)
            from tkinter import messagebox

            messagebox.showerror("WhisperFlow — startup failed", str(exc2), parent=root)
            root.destroy()
            return 1
        first_run_root = root

    if args.headless:
        return run_headless(cfg, ctl, listener)
    return run_with_ui(cfg, ctl, listener, history, autostarted=args.autostart, root=first_run_root)


if __name__ == "__main__":
    raise SystemExit(main())
