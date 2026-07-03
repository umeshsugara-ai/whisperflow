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
import os
import time
from ctypes import wintypes

from whisperflow.config import InjectConfig

from . import clipboard, injector

log = logging.getLogger(__name__)

user32 = ctypes.windll.user32
kernel32 = ctypes.windll.kernel32

VK_MENU = 0x12  # Alt — a lone tap unlocks SetForegroundWindow (never completes a hotkey combo)


def current_window() -> int:
    return user32.GetForegroundWindow()


def window_title(hwnd: int) -> str:
    buf = ctypes.create_unicode_buffer(128)
    user32.GetWindowTextW(hwnd, buf, 128)
    return buf.value


def _window_pid(hwnd: int) -> int:
    pid = wintypes.DWORD(0)
    user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
    return pid.value


def _poll_foreground(hwnd: int, timeout_s: float = 0.3, interval_s: float = 0.02) -> bool:
    """Poll until `hwnd` is foreground or the timeout passes."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if current_window() == hwnd:
            return True
        time.sleep(interval_s)
    return current_window() == hwnd


def try_activate(hwnd: int) -> bool:
    """Best-effort re-activation of the original target window, escalating
    through the tricks Windows requires to bypass the foreground lock."""
    if not hwnd or not user32.IsWindow(hwnd):
        return False

    # attempt 1: plain restore + raise
    user32.ShowWindow(hwnd, 9)  # SW_RESTORE, in case it got minimized
    user32.BringWindowToTop(hwnd)
    user32.SetForegroundWindow(hwnd)
    if _poll_foreground(hwnd):
        return True

    # attempt 2: attach our input queue to the foreground thread's
    fg = current_window()
    fg_thread = user32.GetWindowThreadProcessId(fg, None) if fg else 0
    our_thread = kernel32.GetCurrentThreadId()
    if fg_thread and fg_thread != our_thread:
        attached = user32.AttachThreadInput(our_thread, fg_thread, True)
        try:
            user32.SetForegroundWindow(hwnd)
        finally:
            if attached:
                user32.AttachThreadInput(our_thread, fg_thread, False)
        if _poll_foreground(hwnd):
            return True

    # attempt 3: a lone Alt tap marks our process as "recently received input",
    # which lifts the SetForegroundWindow restriction
    clipboard._send(
        [clipboard._key_event(VK_MENU), clipboard._key_event(VK_MENU, up=True)]
    )
    user32.SetForegroundWindow(hwnd)
    return _poll_foreground(hwnd)


def inject_guarded(text: str, cfg: InjectConfig, expected_hwnd: int) -> str:
    """Inject only if the original target window has focus (restoring it if
    possible). Otherwise put the text on the clipboard and say so."""
    if expected_hwnd:
        cur = current_window()
        if cur != expected_hwnd:
            if _window_pid(cur) == os.getpid():
                # our own pill/tray/window grabbed foreground — always recoverable,
                # SetForegroundWindow is unrestricted while we own the foreground
                log.debug("own window %r has focus — restoring target", window_title(cur))
            else:
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
