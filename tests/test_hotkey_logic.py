"""Unit tests for the pure tap-vs-hold discrimination logic (no real hook)."""

from whisperflow.hotkey import HotkeyEvent, HotkeyStateMachine, format_hotkey_label


def test_format_hotkey_label_common_combos():
    assert format_hotkey_label("alt+windows") == "Alt+Win"
    assert format_hotkey_label("ctrl+windows") == "Ctrl+Win"
    assert format_hotkey_label("windows+space") == "Win+Space"


def test_format_hotkey_label_titlecases_unknown_and_trims():
    assert format_hotkey_label("ctrl + f9") == "Ctrl+F9"
    assert format_hotkey_label("SHIFT+alt") == "Shift+Alt"


def make_sm() -> HotkeyStateMachine:
    return HotkeyStateMachine(tap_threshold_ms=350)


def test_tap_starts_toggle_recording_and_second_tap_stops():
    sm = make_sm()
    t = 100.0
    assert sm.combo_down(t) == HotkeyEvent.RECORD_START
    assert sm.combo_up(t + 0.1) is None  # 100ms < 350ms -> toggle mode, keep recording
    assert sm.recording
    # second tap stops on key-down (instant feel)
    assert sm.combo_down(t + 3.0) == HotkeyEvent.RECORD_STOP
    assert sm.combo_up(t + 3.1) is None  # trailing key-up ignored
    assert not sm.recording


def test_hold_release_stops():
    sm = make_sm()
    t = 200.0
    assert sm.combo_down(t) == HotkeyEvent.RECORD_START
    assert sm.combo_up(t + 1.2) == HotkeyEvent.RECORD_STOP  # 1200ms > 350ms -> hold mode
    assert not sm.recording


def test_hold_with_threshold_callback_then_release():
    sm = make_sm()
    t = 300.0
    assert sm.combo_down(t) == HotkeyEvent.RECORD_START
    sm.hold_threshold_reached()  # timer fired while still held
    assert sm.combo_up(t + 2.0) == HotkeyEvent.RECORD_STOP
    assert not sm.recording


def test_esc_cancels_during_toggle_recording():
    sm = make_sm()
    t = 400.0
    sm.combo_down(t)
    sm.combo_up(t + 0.05)  # tap -> toggle recording
    assert sm.recording
    assert sm.esc() == HotkeyEvent.RECORD_CANCEL
    assert not sm.recording


def test_esc_cancels_during_hold():
    sm = make_sm()
    t = 500.0
    sm.combo_down(t)
    assert sm.esc() == HotkeyEvent.RECORD_CANCEL
    assert not sm.recording
    # subsequent release is a no-op
    assert sm.combo_up(t + 1.0) is None


def test_esc_when_idle_is_noop():
    sm = make_sm()
    assert sm.esc() is None


def test_key_repeat_while_held_is_ignored():
    sm = make_sm()
    t = 600.0
    assert sm.combo_down(t) == HotkeyEvent.RECORD_START
    assert sm.combo_down(t + 0.05) is None  # OS auto-repeat
    assert sm.combo_down(t + 0.10) is None
    assert sm.combo_up(t + 1.0) == HotkeyEvent.RECORD_STOP


def test_exact_threshold_boundary_counts_as_hold():
    sm = HotkeyStateMachine(tap_threshold_ms=350)
    t = 700.0
    sm.combo_down(t)
    # exactly 350ms is NOT under the threshold -> hold semantics
    assert sm.combo_up(t + 0.350) == HotkeyEvent.RECORD_STOP


# --- double-tap-to-start (Wispr-style), enabled via double_tap_ms > 0 ---


def make_dt_sm() -> HotkeyStateMachine:
    return HotkeyStateMachine(tap_threshold_ms=350, double_tap_ms=300)


def test_double_tap_starts_and_keeps_recording_then_single_tap_stops():
    sm = make_dt_sm()
    t = 100.0
    # tap 1
    assert sm.combo_down(t) == HotkeyEvent.RECORD_START
    assert sm.combo_up(t + 0.05) is None  # quick release -> toggle-start
    # tap 2 within the double-tap window -> confirm & KEEP recording (no stop)
    assert sm.combo_down(t + 0.15) is None
    assert sm.combo_up(t + 0.20) is None
    assert sm.recording  # still recording after the fast double-tap
    # a later single tap stops it
    assert sm.combo_down(t + 3.0) == HotkeyEvent.RECORD_STOP
    assert not sm.recording


def test_double_tap_to_stop_does_not_restart():
    sm = make_dt_sm()
    t = 200.0
    sm.combo_down(t)
    sm.combo_up(t + 0.05)
    sm.combo_down(t + 0.15)  # double-tap-to-start
    sm.combo_up(t + 0.20)
    assert sm.recording
    # user double-taps to stop: first tap stops, trailing tap is swallowed
    assert sm.combo_down(t + 3.0) == HotkeyEvent.RECORD_STOP
    assert sm.combo_up(t + 3.05) is None
    assert sm.combo_down(t + 3.15) is None  # swallowed, no phantom restart
    assert not sm.recording


def test_slow_second_tap_stops_like_normal_toggle():
    sm = make_dt_sm()
    t = 300.0
    assert sm.combo_down(t) == HotkeyEvent.RECORD_START
    assert sm.combo_up(t + 0.05) is None  # toggle-start
    # second tap AFTER the double-tap window -> ordinary toggle stop
    assert sm.combo_down(t + 1.0) == HotkeyEvent.RECORD_STOP
    assert not sm.recording


def test_ordinary_toggle_stop_then_quick_restart_is_not_swallowed():
    sm = make_dt_sm()
    t = 350.0
    # ordinary tap-to-start, tap-to-stop -- no double-tap gesture involved
    assert sm.combo_down(t) == HotkeyEvent.RECORD_START
    assert sm.combo_up(t + 0.05) is None  # toggle-start
    assert sm.combo_down(t + 1.0) == HotkeyEvent.RECORD_STOP  # ordinary toggle stop
    # a quick restart within the double-tap window must still start recording
    assert sm.combo_down(t + 1.15) == HotkeyEvent.RECORD_START
    assert sm.recording


def test_hold_to_talk_still_works_with_double_tap_enabled():
    sm = make_dt_sm()
    t = 400.0
    assert sm.combo_down(t) == HotkeyEvent.RECORD_START
    assert sm.combo_up(t + 1.2) == HotkeyEvent.RECORD_STOP  # held -> push-to-talk


# ---- HotkeyListener.rebind (live combo swap, no keyboard hook needed) ----


def test_listener_rebind_swaps_combo_and_resets_chord_state():
    from whisperflow.hotkey import HotkeyListener

    listener = HotkeyListener("ctrl+windows", tap_threshold_ms=350, on_event=lambda e: None)
    assert listener._keys == ["ctrl", "windows"]
    # simulate a half-pressed chord at the moment of the swap
    listener._down_keys.add("ctrl")
    listener._combo_active = True

    listener.rebind("alt+windows")

    assert listener.combo == "alt+windows"
    assert listener._keys == ["alt", "windows"]
    assert listener._down_keys == set()  # stale chord state must not leak
    assert listener._combo_active is False
