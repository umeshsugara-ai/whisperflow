"""System tray icon + menu (pystray, runs on its own thread).

Menu: mode readout · open main window (default) / history / settings ·
cleanup tier radio (live toggle, persisted) · copy last RAW / injected ·
open/reload config · quit.

Callbacks that open windows must NOT touch tkinter here — app.py marshals
them onto the tk main thread via root.after.
"""

from __future__ import annotations

import logging
import os
import threading
from typing import Callable

import pystray
from pystray import Menu, MenuItem

from whisperflow import sysinfo
from whisperflow.config import Config, save_config
from whisperflow.history import History

from . import icons

log = logging.getLogger(__name__)


def _copy_to_clipboard(text: str) -> None:
    from whisperflow.inject import clipboard

    clipboard.write_text(text)


class Tray:
    def __init__(
        self,
        cfg: Config,
        history: History,
        on_reload_config: Callable[[], None],
        on_quit: Callable[[], None],
        on_tier_change: Callable[[str], None],
        on_open_main: Callable[[str], None] = lambda tab: None,
    ) -> None:
        self.cfg = cfg
        self.history = history
        self.on_reload_config = on_reload_config
        self.on_quit = on_quit
        self.on_tier_change = on_tier_change
        self.on_open_main = on_open_main
        self._icons = icons.all_state_icons()
        self._status_line = "idle"

        self.icon = pystray.Icon(
            "WhisperFlow",
            icon=self._icons["idle"],
            title="WhisperFlow — idle",
            menu=self._build_menu(),
        )

    def _build_menu(self) -> Menu:
        def tier_item(tier: str, label: str) -> MenuItem:
            return MenuItem(
                label,
                lambda: self._set_tier(tier),
                checked=lambda item, t=tier: self.cfg.cleanup.tier == t,
                radio=True,
            )

        return Menu(
            MenuItem(lambda item: f"Status: {self._status_line}", None, enabled=False),
            Menu.SEPARATOR,
            MenuItem("Open WhisperFlow", lambda: self.on_open_main("home"), default=True),
            MenuItem("History", lambda: self.on_open_main("history")),
            MenuItem("Settings", lambda: self.on_open_main("settings")),
            Menu.SEPARATOR,
            MenuItem(
                "Cleanup tier",
                Menu(
                    tier_item("off", "Off (verbatim)"),
                    tier_item("rules", "Rules (fillers + punctuation)"),
                    tier_item("llm", "LLM (Ollama, local)"),
                    tier_item("gemini", "LLM (Gemini cloud — text only)"),
                ),
            ),
            MenuItem("Copy last RAW transcript", self._copy_raw),
            MenuItem("Copy last injected text", self._copy_injected),
            Menu.SEPARATOR,
            MenuItem(
                "Start on Windows login",
                self._toggle_autostart,
                checked=lambda item: sysinfo.is_autostart_enabled(),
            ),
            MenuItem("Open config", self._open_config),
            MenuItem("Reload config", lambda: self.on_reload_config()),
            Menu.SEPARATOR,
            MenuItem("Quit WhisperFlow", self._quit),
        )

    def _toggle_autostart(self) -> None:
        if sysinfo.is_autostart_enabled():
            sysinfo.disable_autostart()
        else:
            sysinfo.enable_autostart()

    def _set_tier(self, tier: str) -> None:
        self.cfg.cleanup.tier = tier
        self.on_tier_change(tier)
        try:
            save_config(self.cfg)  # persists — tier used to silently reset on restart
        except Exception as exc:  # noqa: BLE001
            log.warning("could not persist tier change: %s", exc)
        log.info("cleanup tier -> %s", tier)

    def _copy_raw(self) -> None:
        entry = self.history.last()
        if entry:
            _copy_to_clipboard(entry["raw"])

    def _copy_injected(self) -> None:
        entry = self.history.last()
        if entry:
            _copy_to_clipboard(entry["injected"])

    def _open_config(self) -> None:
        os.startfile(str(self.cfg.path))  # noqa: S606

    def _quit(self) -> None:
        self.icon.stop()
        self.on_quit()

    # ---- state updates (thread-safe: pystray handles cross-thread set) ----

    def set_state(self, state_name: str, detail: str = "") -> None:
        icon_key = {
            "IDLE": "idle",
            "RECORDING": "recording",
            "TRANSCRIBING": "processing",
            "INJECTING": "processing",
            "ERROR": "error",
        }.get(state_name, "idle")
        self._status_line = f"{state_name.lower()}{' — ' + detail if detail else ''}"
        self.icon.icon = self._icons[icon_key]
        self.icon.title = f"WhisperFlow — {self._status_line}"[:120]

    def run_detached(self) -> None:
        threading.Thread(target=self.icon.run, daemon=True, name="wf-tray").start()

    def stop(self) -> None:
        self.icon.stop()
