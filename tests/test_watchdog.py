# -*- coding: utf-8 -*-
"""Crash-watchdog decision logic (pure functions — no processes spawned)."""

from whisperflow.watchdog import (
    CRASH_LOOP_LIMIT,
    CRASH_LOOP_WINDOW_S,
    HEARTBEAT_STALE_S,
    build_crash_report,
    crash_loop_exceeded,
    heartbeat_stale,
    read_heartbeat,
    should_relaunch,
)


def test_should_relaunch_only_on_real_crashes():
    assert should_relaunch(1) is True  # python unhandled exception / taskkill
    assert should_relaunch(3221225477) is True  # 0xC0000005 native access violation
    assert should_relaunch(0) is False  # tray Quit / single-instance handoff
    assert should_relaunch(None) is False  # process already gone — don't guess


def test_crash_loop_guard_trips_only_on_consecutive_fast_deaths():
    fast = CRASH_LOOP_WINDOW_S / 2
    slow = CRASH_LOOP_WINDOW_S * 10

    # fewer fast deaths than the limit: keep relaunching
    assert crash_loop_exceeded([fast] * (CRASH_LOOP_LIMIT - 1)) is False
    # limit reached: give up
    assert crash_loop_exceeded([fast] * CRASH_LOOP_LIMIT) is True
    # a long healthy run in between resets the pattern
    assert crash_loop_exceeded([fast, fast, slow, fast]) is False
    # long-lived sessions crashing occasionally: always relaunch
    assert crash_loop_exceeded([slow] * 10) is False


def test_build_crash_report_contains_code_action_and_log_tail():
    report = build_crash_report(
        3221225477, "2026-07-16 21:30:00", "last log line here", relaunching=True
    )
    assert "3221225477" in report
    assert "0xC0000005" in report  # the recognizable native-crash code
    assert "restarted automatically" in report
    assert "last log line here" in report

    stopped = build_crash_report(1, "2026-07-16 21:30:00", "", relaunching=False)
    assert "NOT restarted" in stopped
    assert "crash loop" in stopped


def test_build_crash_report_distinguishes_hang_from_crash():
    hung = build_crash_report(
        1, "2026-07-17 19:00:00", "tail", relaunching=True, was_hang=True
    )
    assert "stopped responding" in hung
    assert "restarted automatically" in hung
    first_line = hung.splitlines()[0]
    assert "crashed" not in first_line  # a freeze must not be reported as a crash

    stopped_hang = build_crash_report(
        1, "2026-07-17 19:00:00", "tail", relaunching=False, was_hang=True
    )
    assert "NOT restarted" in stopped_hang
    assert "freezing" in stopped_hang

    # default (was_hang=False) keeps the original crash wording verbatim
    crashed = build_crash_report(1, "2026-07-17 19:00:00", "tail", relaunching=True)
    assert "crashed" in crashed.splitlines()[0]


def test_read_heartbeat_missing_or_corrupt(tmp_path):
    assert read_heartbeat(tmp_path / "nope.txt") is None

    corrupt = tmp_path / "bad.txt"
    corrupt.write_text("not-a-number", encoding="utf-8")
    assert read_heartbeat(corrupt) is None

    good = tmp_path / "good.txt"
    good.write_text("12345.5", encoding="utf-8")
    assert read_heartbeat(good) == 12345.5


def test_heartbeat_stale_boundary():
    now = 10_000.0
    assert heartbeat_stale(None, now) is False  # no heartbeat yet — never a false hang
    assert heartbeat_stale(now - HEARTBEAT_STALE_S + 1, now) is False  # just inside
    assert heartbeat_stale(now - HEARTBEAT_STALE_S - 1, now) is True  # just outside
    assert heartbeat_stale(now, now) is False  # fresh heartbeat
