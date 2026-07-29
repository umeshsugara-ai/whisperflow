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


# --- the accidental-activation guard: a LONE tap must be discarded ---


def test_lone_tap_is_discarded_when_the_window_lapses():
    """The whole point of double-tap mode: a stray brush of the chord must
    not leave a recording running that later types a stray paragraph."""
    sm = make_dt_sm()
    t = 500.0
    assert sm.combo_down(t) == HotkeyEvent.RECORD_START  # provisional
    assert sm.combo_up(t + 0.05) is None
    assert sm.awaiting_double_tap  # unconfirmed
    assert sm.double_tap_window_expired(t + 0.4) == HotkeyEvent.RECORD_CANCEL
    assert not sm.recording
    assert not sm.awaiting_double_tap
    # and the next lone tap behaves the same, not as a "second tap"
    assert sm.combo_down(t + 2.0) == HotkeyEvent.RECORD_START


def test_expiry_is_a_no_op_once_the_second_tap_confirmed():
    sm = make_dt_sm()
    t = 600.0
    sm.combo_down(t)
    sm.combo_up(t + 0.05)
    sm.combo_down(t + 0.15)  # confirming tap
    sm.combo_up(t + 0.20)
    assert not sm.awaiting_double_tap
    # a timer that fires anyway (already scheduled) must NOT kill the recording
    assert sm.double_tap_window_expired(t + 0.4) is None
    assert sm.recording


def test_expiry_before_the_window_closes_is_ignored():
    """Clock re-check: an early/spurious call must not discard a tap the
    user is still in the middle of completing."""
    sm = make_dt_sm()
    t = 700.0
    sm.combo_down(t)
    sm.combo_up(t + 0.05)
    assert sm.double_tap_window_expired(t + 0.1) is None  # only 50ms in
    assert sm.recording


def test_expiry_never_touches_a_held_or_disabled_session():
    # hold-to-talk: still down, undecided -> not awaiting anything
    sm = make_dt_sm()
    t = 800.0
    sm.combo_down(t)
    assert sm.double_tap_window_expired(t + 0.5) is None
    assert sm.recording

    # feature off entirely -> a lone tap is a normal toggle, never discarded
    off = HotkeyStateMachine(tap_threshold_ms=350, double_tap_ms=0)
    off.combo_down(t)
    off.combo_up(t + 0.05)
    assert off.awaiting_double_tap is False
    assert off.double_tap_window_expired(t + 5.0) is None
    assert off.recording


# --- listener wiring: the confirmation timer tracks the state machine ---


class _FakeTimer:
    """Records arm/cancel instead of really sleeping."""

    instances: list = []

    def __init__(self, delay, fn):
        self.delay, self.fn, self.started, self.cancelled = delay, fn, False, False
        _FakeTimer.instances.append(self)

    def start(self):
        self.started = True

    def cancel(self):
        self.cancelled = True


def test_listener_arms_timer_on_lone_tap_and_cancels_on_confirm(monkeypatch):
    import threading
    import types

    from whisperflow import hotkey as hk
    from whisperflow.hotkey import DEFAULT_DOUBLE_TAP_MS, HotkeyEvent

    _fake_kb(monkeypatch)
    monkeypatch.setattr(hk, "physically_down", lambda name: True)
    _FakeTimer.instances = []
    monkeypatch.setattr(threading, "Timer", _FakeTimer)

    events: list = []
    listener = hk.HotkeyListener(
        "alt+windows", tap_threshold_ms=350, on_event=events.append,
        double_tap_ms=DEFAULT_DOUBLE_TAP_MS,
    )
    down = lambda k: types.SimpleNamespace(name=k, event_type="down")  # noqa: E731
    up = lambda k: types.SimpleNamespace(name=k, event_type="up")  # noqa: E731

    listener._on_key(down("alt"))
    listener._on_key(down("windows"))
    listener._on_key(up("windows"))  # quick release -> provisional toggle
    assert events == [HotkeyEvent.RECORD_START]
    armed = [t for t in _FakeTimer.instances if t.started and not t.cancelled]
    assert len(armed) == 1
    assert armed[0].delay >= DEFAULT_DOUBLE_TAP_MS / 1000.0  # never fires early

    listener._on_key(down("windows"))  # confirming second tap
    assert armed[0].cancelled  # window closed, no discard scheduled
    assert not [t for t in _FakeTimer.instances if t.started and not t.cancelled]


def test_listener_timeout_emits_cancel_for_an_unconfirmed_tap(monkeypatch):
    import threading
    import types

    from whisperflow import hotkey as hk
    from whisperflow.hotkey import DEFAULT_DOUBLE_TAP_MS, HotkeyEvent

    _fake_kb(monkeypatch)
    monkeypatch.setattr(hk, "physically_down", lambda name: True)
    _FakeTimer.instances = []
    monkeypatch.setattr(threading, "Timer", _FakeTimer)
    clock = [1000.0]  # deterministic monotonic clock
    monkeypatch.setattr(hk.time, "monotonic", lambda: clock[0])

    events: list = []
    listener = hk.HotkeyListener(
        "alt+windows", tap_threshold_ms=350, on_event=events.append,
        double_tap_ms=DEFAULT_DOUBLE_TAP_MS,
    )
    listener._on_key(types.SimpleNamespace(name="alt", event_type="down"))
    listener._on_key(types.SimpleNamespace(name="windows", event_type="down"))
    clock[0] += 0.05  # quick release -> provisional toggle-start
    listener._on_key(types.SimpleNamespace(name="windows", event_type="up"))

    timer = [t for t in _FakeTimer.instances if t.started and not t.cancelled][0]
    clock[0] += 0.5  # the window lapses with no second tap
    timer.fn()

    assert events == [HotkeyEvent.RECORD_START, HotkeyEvent.RECORD_CANCEL]
    assert not listener.sm.recording


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


# ---- hook-death watchdog (sleep/resume kills WH_KEYBOARD_LL) ----


def test_probe_due_only_after_idle_window():
    from whisperflow.hotkey import PROBE_IDLE_S, probe_due

    assert probe_due(PROBE_IDLE_S) is True
    assert probe_due(PROBE_IDLE_S + 1) is True
    assert probe_due(PROBE_IDLE_S - 1) is False
    assert probe_due(0.0) is False  # actively typing user is never probed


def test_rearm_swaps_lib_listener_and_reregisters(monkeypatch):
    import sys
    import types

    from whisperflow.hotkey import HotkeyListener

    class FakeListener:
        pass

    fake_kb = types.SimpleNamespace()
    fake_kb.KEY_DOWN = "down"
    fake_kb._listener = FakeListener()
    hooked: list = []
    fake_kb.hook = lambda cb: (hooked.append(cb), cb)[1]
    fake_kb.unhook = lambda h: (_ for _ in ()).throw(KeyError(h))  # dead hook: unhook raises
    monkeypatch.setitem(sys.modules, "keyboard", fake_kb)

    listener = HotkeyListener("alt+windows", tap_threshold_ms=350, on_event=lambda e: None)
    listener._hooks = ["stale-handle"]
    listener._down_keys.add("alt")
    listener._combo_active = True
    old_lib_listener = fake_kb._listener

    listener.rearm()

    # a FRESH keyboard-lib listener was swapped in (forces a new OS hook thread)
    assert fake_kb._listener is not old_lib_listener
    assert isinstance(fake_kb._listener, FakeListener)
    # our handler re-registered; stale handle gone despite unhook raising
    assert hooked == [listener._on_key]
    assert listener._hooks == [listener._on_key]
    # chord state reset — a half-pressed combo from before the death can't leak
    assert listener._down_keys == set()
    assert listener._combo_active is False


def test_on_key_updates_liveness_and_swallows_probe_key(monkeypatch):
    import sys
    import types

    from whisperflow.hotkey import PROBE_KEY, HotkeyListener

    fake_kb = types.SimpleNamespace(KEY_DOWN="down", KEY_UP="up")
    monkeypatch.setitem(sys.modules, "keyboard", fake_kb)

    events: list = []
    listener = HotkeyListener("alt+windows", tap_threshold_ms=350, on_event=events.append)
    listener.last_event_monotonic = 0.0

    probe = types.SimpleNamespace(name=PROBE_KEY, event_type="down")
    listener._on_key(probe)
    assert listener.last_event_monotonic > 0.0  # probe counts as hook liveness
    assert listener._down_keys == set()  # ...but never enters chord state

    down = types.SimpleNamespace(name="alt", event_type="down")
    listener._on_key(down)
    assert "alt" in listener._down_keys  # real keys still work normally


# ---- phantom held keys (a missed key-up must not arm a one-key trigger) ----


def _fake_kb(monkeypatch):
    import sys
    import types

    fake = types.SimpleNamespace(KEY_DOWN="down", KEY_UP="up")
    monkeypatch.setitem(sys.modules, "keyboard", fake)
    return fake


def test_phantom_modifier_does_not_fire_chord_and_self_heals(monkeypatch):
    """Live report: "even pressing Alt alone activates it". Cause: the Win
    key-up was never delivered (UIPI blocks the hook while an elevated
    window has focus), so "windows" stayed in _down_keys and Alt alone
    satisfied the chord."""
    import types

    from whisperflow import hotkey as hk

    _fake_kb(monkeypatch)
    monkeypatch.setattr(hk, "physically_down", lambda name: False)  # OS: nothing held

    events: list = []
    listener = hk.HotkeyListener("alt+windows", tap_threshold_ms=350, on_event=events.append)
    listener._down_keys.add("windows")  # the phantom

    listener._on_key(types.SimpleNamespace(name="alt", event_type="down"))

    assert events == []  # no dictation started
    assert "windows" not in listener._down_keys  # phantom dropped -> self-healed


def test_real_chord_still_fires_when_os_confirms_the_other_key(monkeypatch):
    import types

    from whisperflow import hotkey as hk
    from whisperflow.hotkey import HotkeyEvent

    _fake_kb(monkeypatch)
    monkeypatch.setattr(hk, "physically_down", lambda name: True)  # OS: really held

    events: list = []
    listener = hk.HotkeyListener("alt+windows", tap_threshold_ms=350, on_event=events.append)
    listener._down_keys.add("windows")

    listener._on_key(types.SimpleNamespace(name="alt", event_type="down"))

    assert events == [HotkeyEvent.RECORD_START]


def test_chord_fires_when_key_state_is_unverifiable(monkeypatch):
    """Unmappable key (or no user32) -> physically_down returns None; the
    hook's own bookkeeping is then the only truth and must be trusted."""
    import types

    from whisperflow import hotkey as hk
    from whisperflow.hotkey import HotkeyEvent

    _fake_kb(monkeypatch)
    monkeypatch.setattr(hk, "physically_down", lambda name: None)

    events: list = []
    listener = hk.HotkeyListener("alt+f13", tap_threshold_ms=350, on_event=events.append)
    listener._down_keys.add("f13")

    listener._on_key(types.SimpleNamespace(name="alt", event_type="down"))

    assert events == [HotkeyEvent.RECORD_START]


def test_just_pressed_key_is_never_re_verified(monkeypatch):
    """A low-level hook can see a key-down BEFORE GetAsyncKeyState reflects
    it — re-checking the triggering key would race and kill the hotkey."""
    import types

    from whisperflow import hotkey as hk
    from whisperflow.hotkey import HotkeyEvent

    _fake_kb(monkeypatch)
    asked: list[str] = []

    def spy(name):
        asked.append(name)
        return True

    monkeypatch.setattr(hk, "physically_down", spy)
    events: list = []
    listener = hk.HotkeyListener("alt+windows", tap_threshold_ms=350, on_event=events.append)
    listener._down_keys.add("windows")

    listener._on_key(types.SimpleNamespace(name="alt", event_type="down"))

    assert asked == ["windows"]  # the just-pressed "alt" was NOT queried
    assert events == [HotkeyEvent.RECORD_START]


def test_physically_down_returns_none_for_unmapped_keys():
    from whisperflow.hotkey import physically_down

    assert physically_down("f13") is None
    assert physically_down("") is None
    assert physically_down("alt") in (True, False)  # real OS call, either is fine


# ---- rebind also swaps the double-tap setting (Settings applies live) ----


def test_rebind_updates_double_tap_setting():
    from whisperflow.hotkey import DEFAULT_DOUBLE_TAP_MS, HotkeyListener

    listener = HotkeyListener("alt+windows", tap_threshold_ms=350, on_event=lambda e: None)
    assert listener.sm.double_tap_ms == 0

    listener.rebind("alt+windows", DEFAULT_DOUBLE_TAP_MS)
    assert listener.sm.double_tap_ms == DEFAULT_DOUBLE_TAP_MS

    listener.rebind("alt+windows")  # omitted -> unchanged, not silently reset
    assert listener.sm.double_tap_ms == DEFAULT_DOUBLE_TAP_MS

    listener.rebind("alt+windows", 0)
    assert listener.sm.double_tap_ms == 0
