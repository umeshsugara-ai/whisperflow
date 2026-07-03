"""Tier-2 text injection: clipboard set -> simulated Ctrl+V -> clipboard restore.

Used for long texts (paste is much faster than typing) or as fallback when
unicode typing fails. The user's existing CF_UNICODETEXT clipboard content is
snapshotted before and restored after, so dictation never clobbers a copied
value. Non-text clipboard formats (images, files) cannot be restored — the
injector prefers type-mode by default for exactly that reason.
"""

from __future__ import annotations

import ctypes
import time
from ctypes import wintypes

import win32clipboard
import win32con

user32 = ctypes.WinDLL("user32", use_last_error=True)

INPUT_KEYBOARD = 1
KEYEVENTF_KEYUP = 0x0002
VK_CONTROL = 0x11
VK_V = 0x56


class KEYBDINPUT(ctypes.Structure):
    _fields_ = (
        ("wVk", wintypes.WORD),
        ("wScan", wintypes.WORD),
        ("dwFlags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", ctypes.c_size_t),
    )


class MOUSEINPUT(ctypes.Structure):
    # Largest union member — required for correct sizeof(INPUT) (else error 87).
    _fields_ = (
        ("dx", wintypes.LONG),
        ("dy", wintypes.LONG),
        ("mouseData", wintypes.DWORD),
        ("dwFlags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", ctypes.c_size_t),
    )


class _INPUT_UNION(ctypes.Union):
    _fields_ = (("ki", KEYBDINPUT), ("mi", MOUSEINPUT))


class INPUT(ctypes.Structure):
    _fields_ = (
        ("type", wintypes.DWORD),
        ("union", _INPUT_UNION),
    )


def _key_event(vk: int, up: bool = False) -> INPUT:
    ev = INPUT(type=INPUT_KEYBOARD)
    ev.union.ki = KEYBDINPUT(wVk=vk, wScan=0, dwFlags=KEYEVENTF_KEYUP if up else 0, time=0, dwExtraInfo=0)
    return ev


def _send(events: list[INPUT]) -> None:
    array = (INPUT * len(events))(*events)
    sent = user32.SendInput(len(events), array, ctypes.sizeof(INPUT))
    if sent != len(events):
        raise OSError(f"SendInput injected {sent}/{len(events)} events: {ctypes.get_last_error()}")


def _open_clipboard(retries: int = 5, delay_s: float = 0.05) -> None:
    """Open the clipboard with retries (another app may briefly hold it)."""
    for attempt in range(retries):
        try:
            win32clipboard.OpenClipboard()
            return
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(delay_s)


def read_text() -> str | None:
    """Return current CF_UNICODETEXT clipboard content, or None."""
    _open_clipboard()
    try:
        if win32clipboard.IsClipboardFormatAvailable(win32con.CF_UNICODETEXT):
            return win32clipboard.GetClipboardData(win32con.CF_UNICODETEXT)
        return None
    finally:
        win32clipboard.CloseClipboard()


def write_text(text: str) -> None:
    _open_clipboard()
    try:
        win32clipboard.EmptyClipboard()
        win32clipboard.SetClipboardData(win32con.CF_UNICODETEXT, text)
    finally:
        win32clipboard.CloseClipboard()


def paste_text(text: str, restore_delay_ms: int = 600) -> None:
    """Set clipboard to `text`, simulate Ctrl+V, then restore prior text content.

    restore_delay_ms gives the target app time to read the clipboard before we
    restore — too short and the paste lands with the OLD clipboard content.
    The snapshot is only restored if the clipboard still holds OUR text: if the
    user or another app changed it meanwhile, restoring would clobber theirs.
    """
    snapshot = read_text()
    write_text(text)
    try:
        _send(
            [
                _key_event(VK_CONTROL),
                _key_event(VK_V),
                _key_event(VK_V, up=True),
                _key_event(VK_CONTROL, up=True),
            ]
        )
        time.sleep(restore_delay_ms / 1000.0)
    finally:
        if snapshot is not None:
            try:
                unchanged = read_text() == text
            except Exception:
                unchanged = False
            if unchanged:
                write_text(snapshot)
