"""Crash watchdog — relaunch WhisperFlow when it dies uncleanly OR hangs.

A tiny sibling process (stdlib + ctypes only, no tk/numpy/audio) that the app
spawns after a successful startup. It waits on the app's process handle:

- exit code 0 (tray Quit, single-instance handoff) -> watchdog just exits;
- any other exit (unhandled exception, native crash in a bundled DLL,
  Task-Manager kill) -> write a crash report to <data-dir>/crashes/ and
  relaunch the app, then keep watching the new process;
- process still alive but its heartbeat file has gone stale (Tk mainloop
  frozen — e.g. the app survived an interrupted Windows shutdown in a
  half-initialized state) -> same recovery: kill it, write a report, relaunch.
  A crash only ever catches the app dying; most of what a "stuck, does
  nothing, but looks alive" support report turns out to be is this case.

A crash-loop guard stops relaunching when the app keeps dying right after
start (broken install, poisoned state) — restart-forever would melt the
machine and hide the real problem.

The relaunched app gets WHISPERFLOW_WATCHED=1 so it does NOT spawn a second
watchdog — this one keeps watching. The installer kills the watchdog BEFORE
the app during upgrades/uninstalls, so an intentional taskkill is never
mistaken for a crash to recover from.

Frozen builds ship this as WhisperFlowWatchdog.exe next to WhisperFlow.exe
(same _internal, a few MB); dev runs use `python -m whisperflow.watchdog`.
"""

from __future__ import annotations

import argparse
import ctypes
import os
import subprocess
import sys
import time
from ctypes import wintypes
from pathlib import Path

WATCHED_ENV = "WHISPERFLOW_WATCHED"

# crash-loop guard: an app life shorter than this counts as "died right after
# start"; that many short lives in a row and the watchdog gives up.
CRASH_LOOP_WINDOW_S = 60.0
CRASH_LOOP_LIMIT = 3

LOG_TAIL_LINES = 80

# hang detection: the app touches <data-dir>/heartbeat.txt every
# app.HEARTBEAT_INTERVAL_S via a callback scheduled on the Tk mainloop
# itself — if the mainloop is frozen, the callback never fires and the
# heartbeat goes stale. HEARTBEAT_STALE_S gives ~3 missed ticks of margin
# before that's treated as a hang, not a transient slow tick.
HEARTBEAT_POLL_S = 15.0
HEARTBEAT_STALE_S = 75.0

SYNCHRONIZE = 0x0010_0000
PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
PROCESS_TERMINATE = 0x0001
INFINITE = 0xFFFF_FFFF
WAIT_OBJECT_0 = 0x0000_0000
WAIT_TIMEOUT = 0x0000_0102
DETACHED_PROCESS = 0x0000_0008
CREATE_NEW_PROCESS_GROUP = 0x0000_0200


def should_relaunch(exit_code: int | None) -> bool:
    """Pure decision: relaunch only on a real crash.

    0 = clean quit (tray Quit) or the single-instance "already running"
    handoff; None = the process was already gone before we could open a
    handle (can't tell what happened — don't guess-relaunch next to a
    possibly-running instance; single-instance would kill the newcomer
    anyway, but why churn)."""
    return exit_code is not None and exit_code != 0


def crash_loop_exceeded(life_spans_s: list[float]) -> bool:
    """Pure decision: True when the last CRASH_LOOP_LIMIT app lives were ALL
    shorter than CRASH_LOOP_WINDOW_S — the app is dying at startup and
    relaunching again would loop forever."""
    if len(life_spans_s) < CRASH_LOOP_LIMIT:
        return False
    return all(s < CRASH_LOOP_WINDOW_S for s in life_spans_s[-CRASH_LOOP_LIMIT:])


def build_crash_report(
    exit_code: int, when: str, log_tail: str, relaunching: bool, was_hang: bool = False
) -> str:
    """Pure: the text saved to crashes/crash-<ts>.txt — everything needed to
    debug a teammate's crash without asking them to hunt for logs."""
    kind = "stopped responding" if was_hang else "crashed"
    verb = "freezing" if was_hang else "crashing"
    action = (
        "WhisperFlow was restarted automatically."
        if relaunching
        else f"NOT restarted — it kept {verb} right after starting (crash loop guard)."
    )
    return (
        f"WhisperFlow {kind} — {when}\n"
        f"Exit code: {exit_code} (0x{exit_code & 0xFFFFFFFF:08X})\n"
        f"{action}\n"
        f"\n--- last {LOG_TAIL_LINES} log lines ---\n"
        f"{log_tail}\n"
    )


def read_log_tail(log_path: Path, lines: int = LOG_TAIL_LINES) -> str:
    try:
        text = log_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return "(log file unavailable)"
    return "\n".join(text.splitlines()[-lines:])


def read_heartbeat(path: Path) -> float | None:
    """The timestamp (epoch seconds) the app last wrote, or None if it
    hasn't written one yet (fresh launch/relaunch) or the file is missing
    or corrupt — never a value the caller could mistake for a hang."""
    try:
        return float(path.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return None


def heartbeat_stale(last_beat: float | None, now: float, stale_after_s: float = HEARTBEAT_STALE_S) -> bool:
    """Pure: True only when we HAVE a heartbeat and it's too old. No
    heartbeat yet (fresh launch, older app build without this feature, a
    single missed write) must never read as a hang."""
    if last_beat is None:
        return False
    return (now - last_beat) > stale_after_s


def wait_for_exit_or_hang(pid: int, heartbeat_path: Path) -> tuple[int | None, bool]:
    """Block until the process dies OR its heartbeat goes stale while still
    alive. Returns (exit_code, was_hang). A frozen-but-running process is
    force-terminated so it gets the same crash-report + relaunch recovery
    as a real crash — the caller can't tell the two apart from the outside,
    and a stuck instance is exactly as unusable as a dead one."""
    kernel32 = ctypes.windll.kernel32
    handle = kernel32.OpenProcess(
        SYNCHRONIZE | PROCESS_QUERY_LIMITED_INFORMATION | PROCESS_TERMINATE, False, pid
    )
    if not handle:
        return None, False
    try:
        while True:
            result = kernel32.WaitForSingleObject(handle, int(HEARTBEAT_POLL_S * 1000))
            if result == WAIT_OBJECT_0:
                code = wintypes.DWORD()
                kernel32.GetExitCodeProcess(handle, ctypes.byref(code))
                return code.value, False
            # WAIT_TIMEOUT: still alive — is it still responsive?
            if heartbeat_stale(read_heartbeat(heartbeat_path), time.time()):
                kernel32.TerminateProcess(handle, 1)
                kernel32.WaitForSingleObject(handle, 5000)
                return 1, True
    finally:
        kernel32.CloseHandle(handle)


def relaunch(cmd: list[str]) -> int:
    """Start a fresh app instance, detached, marked as already-watched."""
    env = dict(os.environ)
    env[WATCHED_ENV] = "1"
    proc = subprocess.Popen(  # noqa: S603 — cmd comes from our own spawner
        cmd,
        env=env,
        creationflags=DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP,
        close_fds=True,
        cwd=str(Path(cmd[0]).parent),
    )
    return proc.pid


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pid", type=int, required=True, help="app process id to watch")
    ap.add_argument("--data-dir", required=True, help="WhisperFlow data dir (crash reports + log)")
    # REMAINDER, not nargs="+": the relaunch command carries its own flags
    # (e.g. --config <path>) which a plain positional would reject as
    # unknown options — that exact mistake silently killed the first
    # shipped watchdog at spawn (exit 2, windowed exe, no stderr).
    ap.add_argument("cmd", nargs=argparse.REMAINDER, help="command that relaunches the app")
    args = ap.parse_args()
    if args.cmd and args.cmd[0] == "--":
        args.cmd = args.cmd[1:]
    if not args.cmd:
        ap.error("missing relaunch command")

    data_dir = Path(args.data_dir)
    crashes = data_dir / "crashes"
    heartbeat_path = data_dir / "heartbeat.txt"
    pid = args.pid
    life_spans: list[float] = []

    # a stale heartbeat left over from a previous session must never read as
    # an instant hang the moment this watchdog starts watching a fresh pid
    try:
        heartbeat_path.unlink(missing_ok=True)
    except OSError:
        pass

    while True:
        started = time.monotonic()
        exit_code, was_hang = wait_for_exit_or_hang(pid, heartbeat_path)
        life_spans.append(time.monotonic() - started)
        if not should_relaunch(exit_code):
            return 0

        relaunching = not crash_loop_exceeded(life_spans)
        when = time.strftime("%Y-%m-%d %H:%M:%S")
        report = build_crash_report(
            exit_code,
            when,
            read_log_tail(data_dir / "logs" / "whisperflow.log"),
            relaunching,
            was_hang=was_hang,
        )
        try:
            crashes.mkdir(parents=True, exist_ok=True)
            stamp = time.strftime("%Y%m%d-%H%M%S")
            (crashes / f"crash-{stamp}.txt").write_text(report, encoding="utf-8")
        except OSError:
            pass  # a failed report must not stop the relaunch

        if not relaunching:
            return 1
        time.sleep(1.0)  # let handles/single-instance objects of the dead app clear
        try:
            heartbeat_path.unlink(missing_ok=True)  # grace period for the new process
        except OSError:
            pass
        pid = relaunch(args.cmd)


if __name__ == "__main__":
    raise SystemExit(main())
