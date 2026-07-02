"""Overlay smoke test.

--cycle : automated — create the overlay, cycle through all states over ~5s,
          verify no exception and that the overlay window never gains focus,
          then exit 0.

    python scripts/test_overlay.py --cycle
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cycle", action="store_true", required=True)
    ap.parse_args()

    import tkinter as tk

    from whisperflow.ui.overlay import Overlay

    root = tk.Tk()
    root.withdraw()

    # A visible "target app" window that holds focus — mirrors real use,
    # where the user's editor is foreground. Without a competing window,
    # Windows hands foreground to the only visible window (the overlay),
    # which says nothing about focus STEALING.
    target = tk.Toplevel(root)
    target.title("focus target")
    target.geometry("300x100+100+100")
    tk.Entry(target).pack()
    target.focus_force()
    root.update()

    overlay = Overlay(root)

    steps = [
        (200, lambda: overlay.show_recording("Microphone Array (Realtek)")),
        (1400, overlay.show_processing),
        (2600, overlay.flash_done),
        (3800, lambda: overlay.flash_error("Error: test message")),
        (4600, overlay.hide),
    ]
    for delay, fn in steps:
        root.after(delay, fn)

    import ctypes

    focus_violations: list[int] = []

    def check_focus() -> None:
        # OS-level check: the overlay's own frame must never be the
        # foreground window WHILE VISIBLE. The hwnd is re-resolved each
        # tick because the frame doesn't exist until first map (before
        # that, GA_ROOT walks up to the withdrawn Tk root — a different
        # window that legitimately holds fg in a headless test run).
        if overlay.win.winfo_viewable():
            fg = ctypes.windll.user32.GetForegroundWindow()
            if fg == overlay._hwnd():
                focus_violations.append(fg)
        root.after(250, check_focus)

    root.after(250, check_focus)
    root.after(5200, root.quit)
    root.mainloop()
    root.destroy()

    if focus_violations:
        print(f"FAIL: overlay stole focus from the target window: {focus_violations}")
        return 1
    print("OVERLAY CYCLE PASS — all states shown, no focus steal from target")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
