# -*- coding: utf-8 -*-
"""Crash-watchdog decision logic (pure functions — no processes spawned)."""

from whisperflow.watchdog import (
    CRASH_LOOP_LIMIT,
    CRASH_LOOP_WINDOW_S,
    build_crash_report,
    crash_loop_exceeded,
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
