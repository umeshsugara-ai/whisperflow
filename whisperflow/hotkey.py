"""Global hotkey capture with tap-vs-hold discrimination.

One binding (default Ctrl+Win) serves both trigger modes:
- key-down starts recording immediately;
- released within tap_threshold_ms  -> TOGGLE: keep recording until next tap;
- held longer                        -> HOLD-TO-TALK: release stops+transcribes;
- Esc during a recording             -> CANCEL (recording discarded).

When double_tap_ms > 0 a Wispr-style double-tap gesture is enabled: a second
tap arriving within that window of a fresh toggle-start is read as "confirm and
keep recording" instead of the instant start->stop that a fast double-press
would otherwise produce. A later single tap then stops it, and the trailing tap
of a double-tap-to-stop is swallowed so it does not restart recording. With
double_tap_ms == 0 (the default) behaviour is unchanged.

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

    def combo_down(self, now: float) -> HotkeyEvent | None:
        if self._phase == _Phase.IDLE:
            # swallow the trailing tap of a double-tap-to-stop so it doesn't
            # immediately start a new recording
            if (
                self.double_tap_ms > 0
                and self._stopped_at >= 0.0
                and (now - self._stopped_at) * 1000.0 < self.double_tap_ms
            ):
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
            # stop feels instant; the corresponding key-up is ignored)
            self._phase = _Phase.IDLE
            self._stopped_at = now
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

        name = self._normalize(kb_event.name)
        now = time.monotonic()
        with self._lock:
            if kb_event.event_type == kb.KEY_DOWN:
                if name == "esc":
                    self._emit(self.sm.esc())
                    return
                self._down_keys.add(name)
                if all(k in self._down_keys for k in self._keys):
                    if not self._combo_active:
                        self._combo_active = True
                        self._emit(self.sm.combo_down(now))
            else:  # KEY_UP
                self._down_keys.discard(name)
                if self._combo_active and name in self._keys:
                    self._combo_active = False
                    self._emit(self.sm.combo_up(now))

    def start(self) -> None:
        import keyboard as kb

        self._hooks.append(kb.hook(self._on_key))
        log.info("hotkey listener started: combo=%s", self.combo)

    def stop(self) -> None:
        import keyboard as kb

        for h in self._hooks:
            kb.unhook(h)
        self._hooks.clear()
