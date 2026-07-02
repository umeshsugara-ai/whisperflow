"""WhisperFlow entry point.

Loads config -> loads the STT model once (held in VRAM) -> starts the hotkey
listener + controller worker. Run with --headless for the no-UI pipeline
(tray + overlay wiring arrives with the UI milestone).

    python app.py --headless
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
if sys.stdout:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from whisperflow.audio import Recorder
from whisperflow.config import load_config
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


def setup_logging() -> None:
    from logging.handlers import RotatingFileHandler

    log_dir = APP_ROOT / "logs"
    log_dir.mkdir(exist_ok=True)
    handlers = [
        RotatingFileHandler(
            log_dir / "whisperflow.log", maxBytes=1_000_000, backupCount=3, encoding="utf-8"
        )
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


def build_controller(cfg) -> tuple[Controller, HotkeyListener, History]:
    engine = create_engine(cfg.model)
    engine.load()
    warmup(engine)

    recorder = Recorder(cfg.audio)
    history = History(APP_ROOT / "history.jsonl", max_entries=cfg.history.max_entries)

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

    target = {"hwnd": 0}  # foreground window captured at recording start

    def remember_target() -> None:
        target["hwnd"] = focus.current_window()

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


def run_with_ui(cfg, ctl, listener, history) -> int:
    import tkinter as tk

    from whisperflow.processing import build_processor
    from whisperflow.ui.overlay import Overlay
    from whisperflow.ui.tray import Tray

    root = tk.Tk()
    root.withdraw()  # no main window — tray + overlay only
    from whisperflow.hotkey import HotkeyEvent

    overlay = Overlay(root)
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
                overlay.post(overlay.flash_done, "Copied — focus changed")
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
        rebuild_processor()
        from whisperflow.dictionary import vocabulary_prompt as vp

        ctl.initial_prompt = vp(cfg.dictionary)
        log.info("config reloaded (model/hotkey changes need restart)")

    def on_quit() -> None:
        root.after(0, root.quit)

    def on_open_viewer() -> None:
        from whisperflow.ui.history_view import HistoryViewer

        root.after(0, lambda: HistoryViewer(root, history))

    tray = Tray(cfg, history, on_reload_config, on_quit, on_tier_change, on_open_viewer)
    ctl.on_state = on_state
    ctl.start()
    listener.start()
    tray.run_detached()
    if cfg.overlay.always_visible:
        root.after(0, lambda: overlay.show_idle(hint=cfg.overlay.show_hint))

    log.info("ready — hotkey %s (tap=toggle, hold=push-to-talk, Esc=cancel)", cfg.hotkey.combo)
    try:
        root.mainloop()
    finally:
        listener.stop()
        ctl.shutdown()
        tray.stop()
    return 0


def print_recommendation() -> int:
    from whisperflow import sysinfo
    from whisperflow.config import ModelConfig

    specs = sysinfo.probe()
    has_key = bool(ModelConfig().resolve_api_key())
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
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    if args.recommend:
        if sys.stdout:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        return print_recommendation()

    setup_logging()

    if not acquire_single_instance():
        log.error("WhisperFlow is already running — exiting.")
        print("WhisperFlow is already running.")
        return 2

    cfg = load_config(args.config)
    log.info(
        "config: engine=%s model=%s hotkey=%s cleanup=%s",
        cfg.model.engine,
        cfg.model.cloud_model if cfg.model.engine != "local" else cfg.model.name,
        cfg.hotkey.combo,
        cfg.cleanup.tier,
    )

    from whisperflow import sysinfo

    warning = sysinfo.startup_check(cfg.model, sysinfo.probe())
    if warning:
        log.warning(warning)

    ctl, listener, history = build_controller(cfg)

    if args.headless:
        return run_headless(cfg, ctl, listener)
    return run_with_ui(cfg, ctl, listener, history)


if __name__ == "__main__":
    raise SystemExit(main())
