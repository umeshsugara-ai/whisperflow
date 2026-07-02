"""Tier-1 text injection: SendInput with KEYEVENTF_UNICODE.

Types arbitrary UTF-16 text (including Devanagari and astral-plane chars via
surrogate pairs) as VK_PACKET keyboard events. No clipboard involvement, so
the user's clipboard is never touched and terminals that mishandle synthetic
Ctrl+V (Wispr Flow's documented failure mode) still receive real key events.
"""

from __future__ import annotations

import ctypes
import time
from ctypes import wintypes

user32 = ctypes.WinDLL("user32", use_last_error=True)

INPUT_KEYBOARD = 1
KEYEVENTF_UNICODE = 0x0004
KEYEVENTF_KEYUP = 0x0002

ULONG_PTR = ctypes.POINTER(ctypes.c_ulong)


class KEYBDINPUT(ctypes.Structure):
    _fields_ = (
        ("wVk", wintypes.WORD),
        ("wScan", wintypes.WORD),
        ("dwFlags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", ctypes.c_size_t),
    )


class MOUSEINPUT(ctypes.Structure):
    # Must be present in the union: it is the largest member, and SendInput
    # rejects the call (error 87) if cbSize doesn't match the full union size.
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


def _unicode_events(text: str) -> list[INPUT]:
    """Build key-down + key-up INPUT events for every UTF-16 code unit.

    Newlines are sent as Enter presses (VK_RETURN) because many edit controls
    ignore a bare U+000A unicode packet.
    """
    VK_RETURN = 0x0D
    events: list[INPUT] = []
    for unit in text.replace("\r\n", "\n"):
        if unit == "\n":
            down = INPUT(type=INPUT_KEYBOARD)
            down.union.ki = KEYBDINPUT(wVk=VK_RETURN, wScan=0, dwFlags=0, time=0, dwExtraInfo=0)
            up = INPUT(type=INPUT_KEYBOARD)
            up.union.ki = KEYBDINPUT(wVk=VK_RETURN, wScan=0, dwFlags=KEYEVENTF_KEYUP, time=0, dwExtraInfo=0)
            events.extend((down, up))
            continue
        # Encode this single character to UTF-16 code units (1 for BMP, 2 for astral)
        encoded = unit.encode("utf-16-le")
        for i in range(0, len(encoded), 2):
            scan = int.from_bytes(encoded[i : i + 2], "little")
            down = INPUT(type=INPUT_KEYBOARD)
            down.union.ki = KEYBDINPUT(wVk=0, wScan=scan, dwFlags=KEYEVENTF_UNICODE, time=0, dwExtraInfo=0)
            up = INPUT(type=INPUT_KEYBOARD)
            up.union.ki = KEYBDINPUT(
                wVk=0, wScan=scan, dwFlags=KEYEVENTF_UNICODE | KEYEVENTF_KEYUP, time=0, dwExtraInfo=0
            )
            events.extend((down, up))
    return events


def type_text(text: str, chunk_size: int = 8, interval_ms: int = 5) -> None:
    """Type `text` into the currently focused window via SendInput.

    Sends events in small chunks with a short delay so slower apps
    (terminals, Electron editors) keep up. Raises OSError on API failure.
    """
    if not text:
        return
    events = _unicode_events(text)
    for start in range(0, len(events), chunk_size * 2):  # *2: down+up pairs
        chunk = events[start : start + chunk_size * 2]
        array = (INPUT * len(chunk))(*chunk)
        sent = user32.SendInput(len(chunk), array, ctypes.sizeof(INPUT))
        if sent != len(chunk):
            raise OSError(f"SendInput injected {sent}/{len(chunk)} events: " f"{ctypes.get_last_error()}")
        if interval_ms > 0:
            time.sleep(interval_ms / 1000.0)
