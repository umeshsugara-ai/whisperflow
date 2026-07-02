"""Central controller: state machine + worker queue.

Wires hotkey events to record -> transcribe -> process -> inject, keeping the
GPU work on a single worker thread. UI layers (tray/overlay) subscribe via
`on_state` — the controller itself has no UI dependency, which keeps it
headless-testable with fake recorder/engine/injector.

States: IDLE -> RECORDING -> TRANSCRIBING -> INJECTING -> IDLE
        (RECORDING -> IDLE on cancel; any worker error -> ERROR -> IDLE)
"""

from __future__ import annotations

import logging
import queue
import threading
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable

from whisperflow.audio import Recording
from whisperflow.hotkey import HotkeyEvent

log = logging.getLogger(__name__)


def is_prompt_echo(text: str, initial_prompt: str, max_words: int = 14) -> bool:
    """Detect whisper hallucinating the initial_prompt back as output.

    On short/unclear audio, whisper often returns (an approximation of) the
    vocabulary/seed prompt instead of what was said. If a short transcript's
    words are almost all found in the prompt, treat it as an echo and drop it.
    """
    if not text or not initial_prompt:
        return False
    norm = lambda s: [w.strip(".,!?;:\"'()") for w in s.lower().split()]  # noqa: E731
    words = [w for w in norm(text) if w]
    if not words or len(words) > max_words:
        return False
    prompt_words = set(norm(initial_prompt))
    hits = sum(1 for w in words if w in prompt_words)
    return hits / len(words) >= 0.75


class State(Enum):
    IDLE = auto()
    RECORDING = auto()
    TRANSCRIBING = auto()
    INJECTING = auto()
    ERROR = auto()


# legal transitions (ERROR always resolves to IDLE)
_ALLOWED: dict[State, set[State]] = {
    State.IDLE: {State.RECORDING},
    State.RECORDING: {State.TRANSCRIBING, State.IDLE},  # IDLE on cancel/too-short
    State.TRANSCRIBING: {State.INJECTING, State.IDLE, State.ERROR},  # IDLE on silence/empty
    State.INJECTING: {State.IDLE, State.ERROR},
    State.ERROR: {State.IDLE},
}


@dataclass
class DictationResult:
    raw_text: str
    injected_text: str
    method: str
    language: str
    duration_s: float
    transcribe_seconds: float
    cleanup_tier: str = "off"


@dataclass
class Controller:
    """Coordinates recorder, stt engine, text processor, and injector.

    recorder      : object with start() -> str, stop() -> Recording, cancel()
    engine        : SttEngine (already loaded)
    process_text  : (raw_text, language) -> (final_text, tier) — cleanup +
                    dictionary hook; identity by default (filled in step 7)
    inject_text   : (text) -> method string
    on_state      : (State, detail: str) -> None — UI callback
    on_result     : (DictationResult) -> None — history/log callback
    language      : forced language code or "" for auto
    initial_prompt: vocabulary bias string
    """

    recorder: object
    engine: object
    inject_text: Callable[[str], str]
    process_text: Callable[[str, str], tuple[str, str]] = field(
        default=lambda text, lang: (text, "off")
    )
    on_state: Callable[[State, str], None] = field(default=lambda s, d: None)
    on_result: Callable[[DictationResult], None] = field(default=lambda r: None)
    language: str = ""
    initial_prompt: str = ""

    _state: State = State.IDLE
    _lock: threading.Lock = field(default_factory=threading.Lock)
    _jobs: "queue.Queue" = field(default_factory=queue.Queue)
    _worker: threading.Thread | None = None
    _stopping: bool = False

    @property
    def state(self) -> State:
        return self._state

    def _set_state(self, new: State, detail: str = "") -> None:
        if new is not self._state and new not in _ALLOWED[self._state]:
            log.warning("illegal transition %s -> %s ignored", self._state.name, new.name)
            return
        self._state = new
        try:
            self.on_state(new, detail)
        except Exception:  # UI callback must never kill the pipeline
            log.exception("on_state callback failed")

    # ---- worker ----

    def start(self) -> None:
        self._stopping = False
        self._worker = threading.Thread(target=self._work_loop, daemon=True, name="wf-worker")
        self._worker.start()

    def shutdown(self) -> None:
        self._stopping = True
        self._jobs.put(None)
        if self._worker:
            self._worker.join(timeout=5)

    def _work_loop(self) -> None:
        while not self._stopping:
            job = self._jobs.get()
            if job is None:
                continue
            recording: Recording = job
            try:
                self._process_recording(recording)
            except Exception as exc:  # noqa: BLE001
                log.exception("dictation pipeline failed")
                self._set_state(State.ERROR, str(exc))
                self._set_state(State.IDLE)

    def _process_recording(self, recording: Recording) -> None:
        if recording.too_short or recording.silent:
            reason = "too short" if recording.too_short else "no speech detected"
            log.info("skipping transcription: %s", reason)
            self._set_state(State.IDLE, reason)
            return

        result = self.engine.transcribe(
            recording.samples,
            language=self.language,
            initial_prompt=self.initial_prompt,
        )
        raw_text = result.text
        if not raw_text:
            self._set_state(State.IDLE, "empty transcript")
            return
        from whisperflow.stt.faster_whisper_engine import HINGLISH_SEED

        if is_prompt_echo(raw_text, f"{self.initial_prompt} {HINGLISH_SEED}"):
            log.info("dropped prompt-echo hallucination: %r", raw_text)
            self._set_state(State.IDLE, "no speech (echo filtered)")
            return

        final_text, tier = self.process_text(raw_text, result.language)

        self._set_state(State.INJECTING)
        method = self.inject_text(final_text)

        outcome = DictationResult(
            raw_text=raw_text,
            injected_text=final_text,
            method=method,
            language=result.language,
            duration_s=result.duration_s,
            transcribe_seconds=result.transcribe_seconds,
            cleanup_tier=tier,
        )
        try:
            self.on_result(outcome)
        except Exception:
            log.exception("on_result callback failed")
        self._set_state(State.IDLE, f"injected via {method}")

    # ---- hotkey entry points ----

    def handle_hotkey(self, event: HotkeyEvent) -> None:
        with self._lock:
            if event is HotkeyEvent.RECORD_START:
                if self._state is not State.IDLE:
                    log.debug("RECORD_START ignored in state %s", self._state.name)
                    return
                device = self.recorder.start()
                self._set_state(State.RECORDING, device)

            elif event is HotkeyEvent.RECORD_STOP:
                if self._state is not State.RECORDING:
                    log.debug("RECORD_STOP ignored in state %s", self._state.name)
                    return
                recording = self.recorder.stop()
                self._set_state(State.TRANSCRIBING)
                self._jobs.put(recording)

            elif event is HotkeyEvent.RECORD_CANCEL:
                if self._state is not State.RECORDING:
                    return
                self.recorder.cancel()
                self._set_state(State.IDLE, "cancelled")
