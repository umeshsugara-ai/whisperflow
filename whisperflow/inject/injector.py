"""Tiered text injector — decides how dictated text reaches the focused app.

Policy (config [inject]):
- method = "type":  always SendInput unicode typing (tier 1)
- method = "paste": always clipboard paste with snapshot/restore (tier 2)
- method = "auto":  type for text <= paste_threshold_chars, paste for longer;
                    if typing raises, fall back to paste once.

Wispr Flow ships paste-first and its Windows record shows why that's fragile
(UIPI, WSL, terminal regressions). We invert the order: real unicode key
events first, clipboard only when length makes typing slow.
"""

from __future__ import annotations

import ctypes
import logging
import time

from whisperflow.config import InjectConfig

from . import clipboard, sendinput

log = logging.getLogger(__name__)

# Shift, Ctrl, Alt, LWin, RWin — any of these held during injection turns
# typed characters into accidental shortcuts (Win+V, Alt+letter menus, ...).
_MODIFIER_VKS = (0x10, 0x11, 0x12, 0x5B, 0x5C)


class InjectionError(RuntimeError):
    """All applicable injection tiers failed."""


def modifiers_down() -> bool:
    """True while any modifier key is physically held — used both to delay
    injection here and by the controller to hold back LIVE chunk injection
    during hold-to-talk (injecting under a held Win/Alt would fire shortcuts)."""
    user32 = ctypes.windll.user32
    return any(user32.GetAsyncKeyState(vk) & 0x8000 for vk in _MODIFIER_VKS)


_modifiers_down = modifiers_down  # backwards-compatible private alias


def _wait_modifiers_released(timeout_ms: int) -> None:
    """Block until the user physically releases all modifier keys.

    The hotkey combo (e.g. Alt+Win) is often still held when a toggle-stop
    fires; injecting under held modifiers drops or reinterprets keystrokes.
    """
    if timeout_ms <= 0:
        return
    deadline = time.monotonic() + timeout_ms / 1000.0
    while _modifiers_down():
        if time.monotonic() >= deadline:
            log.warning(
                "modifier keys still held after %.1fs — injecting anyway", timeout_ms / 1000.0
            )
            return
        time.sleep(0.015)


def inject(text: str, cfg: InjectConfig) -> str:
    """Inject `text` into the focused window. Returns the method used."""
    if not text:
        return "none"

    _wait_modifiers_released(cfg.modifier_release_timeout_ms)

    if cfg.method == "type":
        sendinput.type_text(text, interval_ms=cfg.type_interval_ms)
        return "type"

    if cfg.method == "paste":
        clipboard.paste_text(text, restore_delay_ms=cfg.clipboard_restore_delay_ms)
        return "paste"

    # auto
    if len(text) <= cfg.paste_threshold_chars:
        try:
            sendinput.type_text(text, interval_ms=cfg.type_interval_ms)
            return "type"
        except OSError as exc:
            log.warning("type-injection failed (%s); falling back to paste", exc)
            clipboard.paste_text(text, restore_delay_ms=cfg.clipboard_restore_delay_ms)
            return "paste-fallback"
    else:
        clipboard.paste_text(text, restore_delay_ms=cfg.clipboard_restore_delay_ms)
        return "paste"
