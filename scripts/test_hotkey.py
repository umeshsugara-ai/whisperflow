"""Manual hotkey test — prints events for real key presses.

Run, then try: quick tap of Ctrl+Win (toggle on) -> tap again (toggle off);
hold Ctrl+Win >350ms and release (hold-to-talk); Esc mid-recording (cancel).
Type normally in another window to confirm no false triggers. Ctrl+C to exit.

    python scripts/test_hotkey.py [--combo ctrl+windows]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from whisperflow.hotkey import HotkeyEvent, HotkeyListener  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--combo", default="ctrl+windows")
    ap.add_argument("--tap-ms", type=int, default=350)
    args = ap.parse_args()

    def on_event(ev: HotkeyEvent) -> None:
        stamp = time.strftime("%H:%M:%S")
        print(f"[{stamp}] {ev.name}")

    listener = HotkeyListener(args.combo, args.tap_ms, on_event)
    listener.start()
    print(f"Listening for '{args.combo}' (tap<{args.tap_ms}ms=toggle, hold=push-to-talk, Esc=cancel).")
    print("Ctrl+C to exit.")
    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        listener.stop()
        print("stopped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
