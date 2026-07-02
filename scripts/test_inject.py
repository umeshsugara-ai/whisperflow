"""Injection test harness.

--self-test : automated — opens a local tkinter Text widget, focuses it,
              injects each fixture in type and paste modes, and compares the
              widget content with the fixture. Also asserts the clipboard is
              unchanged (type) / restored (paste). Prints SELF-TEST PASS/FAIL.
--countdown : manual — 3-second countdown, then injects the chosen fixture
              into whatever window you focused (Notepad, Chrome, terminal...).

Usage (from D:\\whisperFlowMy):
    python scripts/test_inject.py --self-test
    python scripts/test_inject.py --countdown [--fixture N] [--mode type|paste]
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Windows consoles default to cp1252 which can't print Devanagari fixtures
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from whisperflow.inject import clipboard, sendinput  # noqa: E402

FIXTURES = [
    ("english", "The quick brown fox jumps over the lazy dog."),
    ("devanagari", "नमस्ते दुनिया, यह एक परीक्षण है"),
    ("hinglish", "Kal meeting hai na, toh please deck ready rakhna yaar."),
    ("symbols", 'it\'s "quoted" — em-dash &<tags> 100% #done'),
]

CLIPBOARD_SENTINEL = "WHISPERFLOW-CLIPBOARD-SENTINEL-12345"


def self_test() -> int:
    import tkinter as tk

    failures: list[str] = []

    root = tk.Tk()
    root.title("WhisperFlow inject self-test")
    root.geometry("600x120+200+200")
    text = tk.Text(root, font=("Nirmala UI", 11))
    text.pack(fill="both", expand=True)
    root.update()

    # Pre-set a clipboard sentinel so we can verify restore/no-clobber
    clipboard.write_text(CLIPBOARD_SENTINEL)

    for mode in ("type", "paste"):
        for name, fixture in FIXTURES:
            text.delete("1.0", "end")
            text.focus_force()
            root.update()
            root.after(50)
            root.update()
            time.sleep(0.15)  # let focus settle

            # Inject from a background thread while the main thread pumps the
            # tk event loop — a real target app has its own live message pump;
            # blocking this one during injection would test an unrealistic app.
            error: list[Exception] = []

            def do_inject() -> None:
                try:
                    if mode == "type":
                        sendinput.type_text(fixture, interval_ms=2)
                    else:
                        clipboard.paste_text(fixture, restore_delay_ms=250)
                except Exception as exc:  # noqa: BLE001
                    error.append(exc)

            worker = threading.Thread(target=do_inject, daemon=True)
            worker.start()

            deadline = time.time() + 3.0
            landed = ""
            while time.time() < deadline:
                root.update()
                landed = text.get("1.0", "end-1c")
                if landed == fixture and not worker.is_alive():
                    break
                time.sleep(0.02)
            worker.join(timeout=2.0)

            if error:
                failures.append(f"{mode}/{name}: injection raised {error[0]!r}")
                continue

            if landed != fixture:
                failures.append(f"{mode}/{name}: expected {fixture!r}, got {landed!r}")

            clip_now = clipboard.read_text()
            if clip_now != CLIPBOARD_SENTINEL:
                failures.append(f"{mode}/{name}: clipboard not preserved (got {clip_now!r})")
                clipboard.write_text(CLIPBOARD_SENTINEL)  # re-arm for next case

    root.destroy()

    if failures:
        print("SELF-TEST FAIL")
        for f in failures:
            print("  -", f)
        return 1
    print(f"SELF-TEST PASS — {len(FIXTURES)} fixtures x 2 modes, clipboard preserved")
    return 0


def countdown(fixture_idx: int, mode: str) -> int:
    name, fixture = FIXTURES[fixture_idx]
    print(f"Focus your target window now. Injecting fixture '{name}' via {mode} in:")
    for i in (3, 2, 1):
        print(f"  {i}...")
        time.sleep(1)
    if mode == "paste":
        clipboard.paste_text(fixture)
    else:
        sendinput.type_text(fixture, interval_ms=5)
    print("Injected. Verify the text landed verbatim:")
    print(f"  {fixture}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--self-test", action="store_true")
    group.add_argument("--countdown", action="store_true")
    ap.add_argument("--fixture", type=int, default=0, help=f"0..{len(FIXTURES) - 1}")
    ap.add_argument("--mode", choices=("type", "paste"), default="type")
    args = ap.parse_args()

    if args.self_test:
        return self_test()
    return countdown(args.fixture, args.mode)


if __name__ == "__main__":
    raise SystemExit(main())
