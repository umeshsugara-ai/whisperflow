"""Pure mapping from pipeline outcomes to pill messages.

Kept free of tkinter so it's unit-testable headless. The pill label truncates
at 28 chars (overlay.flash_warn/flash_error) — keep messages within that.
"""

from __future__ import annotations


def idle_flash(detail: str) -> tuple[str, str] | None:
    """(kind, message) to flash when the pipeline lands back on IDLE, or None
    for the silent default (just show the resting pill).

    kind is "done" | "warn" — matched to overlay.flash_done/flash_warn.
    A recording discarded as silence used to collapse the pill with NO
    feedback at all, which reads as "the app is broken"; now it says why.
    """
    if "clipboard" in detail:
        return ("warn", "Copied — press Ctrl+V")
    if detail.startswith("injected"):
        return ("done", "Injected ✓")
    if "no speech" in detail or "empty transcript" in detail:
        return ("warn", "No speech — check mic ⚠")
    if "too short" in detail:
        return ("warn", "Too short — hold & speak")
    return None
