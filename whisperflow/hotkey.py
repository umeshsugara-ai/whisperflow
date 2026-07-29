"""Global hotkey capture with tap-vs-hold discrimination.

One binding (default Ctrl+Win) serves both trigger modes:
- key-down starts recording immediately;
- released within tap_threshold_ms  -> TOGGLE: keep recording until next tap;
- held longer                        -> HOLD-TO-TALK: release stops+transcribes;
- Esc during a recording             -> CANCEL (recording discarded).

When double_tap_ms > 0 the combo must be tapped TWICE to start dictation: the
first tap starts capturing provisionally, and unless a confirming second tap
arrives within that window the recording is discarded (nothing is transcribed
or typed). This is the guard against accidental activation — a stray brush of
the chord costs a brief pill blink, not a stray paragraph. A later single tap
stops a confirmed recording, and the trailing tap of a double-tap-to-stop is
swallowed so it does not restart recording. Hold-to-talk is unaffected: holding
past tap_threshold_ms records and transcribes on release, no second tap needed.
With double_tap_ms == 0 (the default) a single tap starts as usual.

The timing logic lives in `HotkeyStateMachine` as a pure, clock-injectable
class so it is unit-testable without a real keyboard hook. `HotkeyListener`
wires it to the `keyboard` library (WH_KEYBOARD_LL under the hood).
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from enum import Enum, auto
from typing import Callable

log = logging.getLogger(__name__)


_LABEL_TOKENS = {
    "windows": "Win",
    "win": "Win",
    "cmd": "Win",
    "ctrl": "Ctrl",
    "control": "Ctrl",
    "alt": "Alt",
    "shift": "Shift",
    "space": "Space",
}


def format_hotkey_label(combo: str) -> str:
    """Human-friendly hotkey label from a combo string, for the overlay pill.

    'alt+windows' -> 'Alt+Win', 'ctrl+windows' -> 'Ctrl+Win',
    'windows+space' -> 'Win+Space'. Unknown tokens are title-cased.
    """
    parts = [p.strip().lower() for p in combo.split("+") if p.strip()]
    return "+".join(_LABEL_TOKENS.get(p, p.title()) for p in parts)


# Physical key-state verification. The listener tracks held keys in an
# accumulated set, so ONE missed key-up leaves a phantom modifier "held"
# forever — after which the other combo key ALONE satisfies the chord and
# fires dictation (live report: "even pressing Alt alone activates it").
# Missed key-ups are routine, not exotic: while an elevated window has
# focus, UIPI blocks our low-level hook entirely, so the release of a key
# pressed just before that window appeared never reaches us. GetAsyncKeyState
# is the OS's own answer to "is this key really down right now".
_VK_BY_NAME = {
    "alt": (0x12,),  # VK_MENU — either side
    "ctrl": (0x11,),  # VK_CONTROL
    "control": (0x11,),
    "shift": (0x10,),  # VK_SHIFT
    "windows": (0x5B, 0x5C),  # VK_LWIN / VK_RWIN — no combined code exists
    "space": (0x20,),
}
_KEY_DOWN_BIT = 0x8000

# What the Settings "require a double-tap" checkbox writes into
# [hotkey].double_tap_ms. Wide enough for a comfortable human double-tap,
# short enough that two deliberate separate taps aren't merged into one.
DEFAULT_DOUBLE_TAP_MS = 300


def physically_down(name: str) -> bool | None:
    """True/False per the OS, or None when the key isn't mappable (then the
    hook's own bookkeeping is all we have and the caller should trust it)."""
    vks = _VK_BY_NAME.get(name)
    if not vks:
        return None
    try:
        import ctypes

        get_state = ctypes.windll.user32.GetAsyncKeyState
    except (AttributeError, OSError):  # non-Windows / no user32 — can't verify
        return None
    return any(get_state(vk) & _KEY_DOWN_BIT for vk in vks)


class HotkeyEvent(Enum):
    RECORD_START = auto()  # begin capturing audio
    RECORD_STOP = auto()  # stop and transcribe
    RECORD_CANCEL = auto()  # stop and discard


class _Phase(Enum):
    IDLE = auto()
    DOWN_UNDECIDED = auto()  # combo pressed, tap-vs-hold not yet known
    HOLD_RECORDING = auto()  # held past threshold: hold-to-talk
    TOGGLE_RECORDING = auto()  # tapped: recording until next tap


@dataclass
class HotkeyStateMachine:
    """Pure discrimination logic. Feed it combo_down/combo_up/esc with
    timestamps (seconds); it returns the HotkeyEvent to emit (or None)."""

    tap_threshold_ms: int = 350
    double_tap_ms: int = 0  # 0 = disabled; >0 enables double-tap-to-start
    _phase: _Phase = _Phase.IDLE
    _down_at: float = 0.0
    _toggle_at: float = 0.0  # when the current toggle recording began
    _dt_locked: bool = False  # a double-tap has confirmed this recording
    _stopped_at: float = -1.0  # last stop time; <0 means none recently

    @property
    def phase(self) -> _Phase:
        return self._phase

    @property
    def recording(self) -> bool:
        return self._phase in (_Phase.DOWN_UNDECIDED, _Phase.HOLD_RECORDING, _Phase.TOGGLE_RECORDING)

    @property
    def awaiting_double_tap(self) -> bool:
        """A provisional tap-start still waiting for its confirming second
        tap — the listener arms a timer on this and discards if it lapses."""
        return (
            self.double_tap_ms > 0
            and self._phase is _Phase.TOGGLE_RECORDING
            and not self._dt_locked
        )

    def double_tap_window_expired(self, now: float) -> HotkeyEvent | None:
        """The confirming second tap never came, so the first one was
        accidental: discard the audio. Deliberately re-checks the clock
        rather than trusting the timer — a late/spurious call must not kill
        a recording that was confirmed in the meantime."""
        if not self.awaiting_double_tap:
            return None
        if (now - self._toggle_at) * 1000.0 < self.double_tap_ms:
            return None
        self._phase = _Phase.IDLE
        return HotkeyEvent.RECORD_CANCEL

    def combo_down(self, now: float) -> HotkeyEvent | None:
        if self._phase == _Phase.IDLE:
            # swallow the trailing tap of a double-tap-to-stop so it doesn't
            # immediately start a new recording
            if (
                self.double_tap_ms > 0
                and self._stopped_at >= 0.0
                and (now - self._stopped_at) * 1000.0 < self.double_tap_ms
            ):
                log.debug("combo press swallowed: trailing tap of double-tap-stop")
                self._stopped_at = -1.0
                return None
            self._phase = _Phase.DOWN_UNDECIDED
            self._down_at = now
            return HotkeyEvent.RECORD_START
        if self._phase == _Phase.TOGGLE_RECORDING:
            if self._dt_locked:
                # already double-tap-confirmed: a tap now stops it
                self._phase = _Phase.IDLE
                self._dt_locked = False
                self._stopped_at = now
                return HotkeyEvent.RECORD_STOP
            if self.double_tap_ms > 0 and (now - self._toggle_at) * 1000.0 < self.double_tap_ms:
                # second tap arrived quickly -> double-tap gesture: keep recording
                self._dt_locked = True
                return None
            # second tap ends a toggle recording (handled on key-down so the
            # stop feels instant; the corresponding key-up is ignored). This
            # was never double-tap-confirmed, so there's no trailing tap to
            # guard against -- don't arm _stopped_at, or a quick legitimate
            # restart right after would be silently swallowed.
            self._phase = _Phase.IDLE
            return HotkeyEvent.RECORD_STOP
        # key-repeat while held: ignore
        return None

    def combo_up(self, now: float) -> HotkeyEvent | None:
        if self._phase == _Phase.DOWN_UNDECIDED:
            held_ms = (now - self._down_at) * 1000.0
            if held_ms < self.tap_threshold_ms:
                self._phase = _Phase.TOGGLE_RECORDING  # tap -> keep recording
                self._toggle_at = now
                return None
            self._phase = _Phase.IDLE  # hold -> release stops
            return HotkeyEvent.RECORD_STOP
        if self._phase == _Phase.HOLD_RECORDING:
            self._phase = _Phase.IDLE
            return HotkeyEvent.RECORD_STOP
        # key-up right after a toggle-ending tap, or spurious: ignore
        return None

    def hold_threshold_reached(self) -> None:
        """Optional: called by a timer once threshold passes while still down,
        so phase reflects hold-mode (no event emitted; recording continues)."""
        if self._phase == _Phase.DOWN_UNDECIDED:
            self._phase = _Phase.HOLD_RECORDING

    def esc(self) -> HotkeyEvent | None:
        if self.recording:
            self._phase = _Phase.IDLE
            self._dt_locked = False
            return HotkeyEvent.RECORD_CANCEL
        return None


# Hook-health watchdog tuning. Windows silently destroys WH_KEYBOARD_LL hooks
# (sleep/resume is the classic trigger; hook-timeout evictions and explorer
# restarts do it too) — the app then looks alive but never sees a key again.
# The watchdog notices "no key events for a while", injects an inert probe key
# (F24 — produces no character, virtually nothing binds it) and, if even the
# probe isn't seen, tears the hook down and re-installs it.
PROBE_IDLE_S = 120.0  # any real key event within this window = hook is alive
PROBE_KEY = "f24"  # what the hook reports the probe as
PROBE_VK = 0x87  # VK_F24 — injected via raw SendInput
PROBE_WAIT_S = 1.0  # how long the injected probe may take to reach our handler
WATCHDOG_TICK_S = 30.0


def probe_due(idle_s: float, threshold_s: float = PROBE_IDLE_S) -> bool:
    """Pure decision: probe only after a real silence window, so an actively
    typing user never gets probe injections."""
    return idle_s >= threshold_s


class HotkeyListener:
    """Binds HotkeyStateMachine to real keyboard events via `keyboard` lib."""

    def __init__(
        self,
        combo: str,
        tap_threshold_ms: int,
        on_event: Callable[[HotkeyEvent], None],
        double_tap_ms: int = 0,
    ) -> None:
        self.combo = combo
        self.on_event = on_event
        self.sm = HotkeyStateMachine(
            tap_threshold_ms=tap_threshold_ms, double_tap_ms=double_tap_ms
        )
        self._lock = threading.Lock()
        self._keys = [k.strip().lower() for k in combo.split("+") if k.strip()]
        self._down_keys: set[str] = set()
        self._combo_active = False
        self._hooks: list = []
        self.last_event_monotonic: float = 0.0  # ANY key seen by our handler
        self._watch_stop = threading.Event()
        self._watch_thread: threading.Thread | None = None
        self._dt_timer: threading.Timer | None = None  # double-tap confirmation window

    # `keyboard` lib normalizes "windows" as "windows"/"left windows" etc.
    @staticmethod
    def _normalize(name: str) -> str:
        name = (name or "").lower()
        for prefix in ("left ", "right "):
            if name.startswith(prefix):
                name = name[len(prefix) :]
        if name in ("windows", "win", "cmd"):
            return "windows"
        return name

    def _emit(self, event: HotkeyEvent | None) -> None:
        if event is not None:
            log.debug("hotkey event: %s", event.name)
            self.on_event(event)

    def _on_key(self, kb_event) -> None:
        import keyboard as kb  # local import keeps module importable headless

        now = time.monotonic()
        self.last_event_monotonic = now  # hook liveness (single float write: atomic)
        name = self._normalize(kb_event.name)
        if name == PROBE_KEY:
            return  # our own watchdog probe — not user input
        with self._lock:
            if kb_event.event_type == kb.KEY_DOWN:
                if name == "esc":
                    self._emit(self.sm.esc())
                    self._sync_double_tap_timer()
                    return
                self._down_keys.add(name)
                if all(k in self._down_keys for k in self._keys):
                    if not self._combo_active and self._chord_is_real(name):
                        self._combo_active = True
                        self._emit(self.sm.combo_down(now))
                        self._sync_double_tap_timer()
            else:  # KEY_UP
                self._down_keys.discard(name)
                if self._combo_active and name in self._keys:
                    self._combo_active = False
                    self._emit(self.sm.combo_up(now))
                    self._sync_double_tap_timer()

    def _sync_double_tap_timer(self) -> None:
        """Keep the confirmation timer in step with the state machine: armed
        exactly while a provisional tap-start is waiting, cancelled the moment
        it is confirmed, stopped, or cancelled. Callers hold self._lock."""
        if self._dt_timer is not None:
            self._dt_timer.cancel()
            self._dt_timer = None
        if not self.sm.awaiting_double_tap:
            return
        # small grace so the timer never fires a hair EARLY and gets rejected
        # by double_tap_window_expired's own clock re-check
        delay = self.sm.double_tap_ms / 1000.0 + 0.02
        self._dt_timer = threading.Timer(delay, self._on_double_tap_timeout)
        self._dt_timer.daemon = True
        self._dt_timer.start()

    def _on_double_tap_timeout(self) -> None:
        with self._lock:
            self._dt_timer = None
            self._emit(self.sm.double_tap_window_expired(time.monotonic()))

    def _chord_is_real(self, just_pressed: str) -> bool:
        """Guard against phantom held keys: ask the OS whether the OTHER
        combo keys are actually down. The key that just fired this event is
        trusted as-is — a low-level hook can see a key-down before
        GetAsyncKeyState reflects it, and re-checking it here would be a race
        that silently kills the hotkey. Any key the OS says is NOT held is
        dropped from the set, so the phantom self-heals instead of arming a
        one-key trigger forever."""
        stale = [
            k for k in self._keys
            if k != just_pressed and physically_down(k) is False
        ]
        if not stale:
            return True
        for k in stale:
            self._down_keys.discard(k)
        log.debug("phantom held key(s) dropped, chord not fired: %s", stale)
        return False

    def rebind(self, combo: str, double_tap_ms: int | None = None) -> None:
        """Live-swap the combo (Settings saves apply immediately, no restart).
        The keyboard hook itself is combo-agnostic — only the key set the
        handler matches against changes, so no unhook/rehook is needed."""
        with self._lock:
            self.combo = combo
            self._keys = [k.strip().lower() for k in combo.split("+") if k.strip()]
            self._down_keys.clear()
            self._combo_active = False
            if double_tap_ms is not None:
                self.sm.double_tap_ms = double_tap_ms
            self._sync_double_tap_timer()  # e.g. double-tap just switched off
        log.info(
            "hotkey listener rebound: combo=%s double_tap_ms=%s",
            combo, self.sm.double_tap_ms,
        )

    def rearm(self) -> None:
        """Tear the OS-level hook down and re-install it.

        The `keyboard` lib starts its WH_KEYBOARD_LL hook thread exactly once
        (module-global `_listener`, `listening` latches True) and has no
        public recovery path once Windows destroys the hook — so a fresh
        listener object is swapped in, forcing hook() to spin up a new hook
        thread. The old thread, if still alive, keeps running with zero
        handlers and dispatches nothing (no double events).
        """
        import keyboard as kb

        for h in self._hooks:
            try:
                kb.unhook(h)
            except (KeyError, ValueError):
                pass  # already gone — exactly the state we're recovering from
        self._hooks.clear()
        if hasattr(kb, "_listener"):
            kb._listener = type(kb._listener)()  # noqa: SLF001 — no public API for this
        with self._lock:
            self._down_keys.clear()
            self._combo_active = False
        self._hooks.append(kb.hook(self._on_key))
        log.info("keyboard hook re-armed (combo=%s)", self.combo)

    def _probe_hook_alive(self) -> bool:
        """Inject the inert probe key and check our handler saw it.

        Injection goes through raw SendInput with wVk=F24 (reusing the
        injector's INPUT plumbing) — the `keyboard` lib's own
        press_and_release("f24") silently never reaches the hook (its
        scan-code mapping is broken for high F-keys; verified live), while
        the raw VK event comes back as a normal ('f24', down/up) pair."""
        from whisperflow.inject.clipboard import _key_event, _send

        sent_at = time.monotonic()
        try:
            _send([_key_event(PROBE_VK), _key_event(PROBE_VK, up=True)])
        except OSError:
            log.exception("hook probe injection failed")
            return True  # can't probe -> don't thrash the hook on bad evidence
        deadline = sent_at + PROBE_WAIT_S
        while time.monotonic() < deadline:
            if self.last_event_monotonic >= sent_at:
                return True
            time.sleep(0.05)
        return False

    def _watchdog(self) -> None:
        """Detect a dead hook (sleep/resume is the classic killer) and re-arm.

        Cheap when healthy: while the user types, last_event_monotonic stays
        fresh and the loop does nothing. Only after PROBE_IDLE_S of total key
        silence does it inject one F24 probe to distinguish "user is idle"
        from "hook is dead"."""
        fails = 0  # consecutive dead-probe ticks: log loudly once, then quietly
        while not self._watch_stop.wait(WATCHDOG_TICK_S):
            idle_s = time.monotonic() - self.last_event_monotonic
            if not probe_due(idle_s):
                fails = 0
                continue
            if self._probe_hook_alive():
                if fails:
                    log.info("hook re-arm verified — hotkey is live again")
                fails = 0
                continue
            # A focused elevated (admin) window also blocks our probe (UIPI) —
            # indistinguishable from a dead hook, so re-arming stays correct
            # (it's cheap and harmless), but only the first tick logs loudly.
            level = log.warning if fails == 0 else log.debug
            level(
                "keyboard hook unresponsive (no events, probe unseen) — re-arming. "
                "Windows kills low-level hooks across sleep/resume."
            )
            fails += 1
            try:
                self.rearm()
            except Exception:  # noqa: BLE001
                log.exception("hook re-arm failed — will retry on the next tick")

    def start(self) -> None:
        import keyboard as kb

        self._hooks.append(kb.hook(self._on_key))
        self.last_event_monotonic = time.monotonic()  # arm the idle clock from "now"
        self._watch_stop.clear()
        self._watch_thread = threading.Thread(
            target=self._watchdog, daemon=True, name="wf-hook-watchdog"
        )
        self._watch_thread.start()
        log.info("hotkey listener started: combo=%s", self.combo)

    def stop(self) -> None:
        import keyboard as kb

        self._watch_stop.set()
        with self._lock:
            if self._dt_timer is not None:
                self._dt_timer.cancel()
                self._dt_timer = None
        for h in self._hooks:
            kb.unhook(h)
        self._hooks.clear()
