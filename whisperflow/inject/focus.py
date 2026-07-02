"""Focus guard — never inject into the wrong window.

The target window is captured when recording STARTS. If the user alt-tabs
away before transcription finishes, we first try to re-activate the original
window; if Windows refuses (foreground-lock rules), we do NOT paste blindly —
the text goes to the clipboard instead and the caller reports it clearly.
This directly fixes the "transcript pasted into some unknown window" failure.
"""

from __future__ import annotations

import ctypes
import logging
import time

from whisperflow.config import InjectConfig

from . import clipboard, injector

log = logging.getLogger(__name__)

user32 = ctypes.windll.user32


def current_window() -> int:
    return user32.GetForegroundWindow()


def window_title(hwnd: int) -> str:
    buf = ctypes.create_unicode_buffer(128)
    user32.GetWindowTextW(hwnd, buf, 128)
    return buf.value


def try_activate(hwnd: int) -> bool:
    """Best-effort re-activation of the original target window."""
    if not hwnd or not user32.IsWindow(hwnd):
        return False
    user32.ShowWindow(hwnd, 9)  # SW_RESTORE, in case it got minimized
    user32.SetForegroundWindow(hwnd)
    time.sleep(0.12)  # let the focus change settle before typing
    return current_window() == hwnd


def inject_guarded(text: str, cfg: InjectConfig, expected_hwnd: int) -> str:
    """Inject only if the original target window has focus (restoring it if
    possible). Otherwise put the text on the clipboard and say so."""
    if expected_hwnd:
        cur = current_window()
        if cur != expected_hwnd:
            log.info(
                "focus changed (was %r, now %r) — attempting to restore",
                window_title(expected_hwnd),
                window_title(cur),
            )
            if not try_activate(expected_hwnd):
                clipboard.write_text(text)
                log.warning("could not restore focus — text copied to clipboard instead")
                return "clipboard (focus changed)"
    return injector.inject(text, cfg)
