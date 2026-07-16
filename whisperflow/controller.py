"""Central controller: state machine + worker queue.

Wires hotkey events to record -> transcribe -> process -> inject, keeping the
GPU work on a single worker thread. UI layers (tray/overlay) subscribe via
`on_state` — the controller itself has no UI dependency, which keeps it
headless-testable with fake recorder/engine/injector.

States: IDLE -> RECORDING -> TRANSCRIBING -> INJECTING -> IDLE
        (RECORDING -> IDLE on cancel; any worker error -> ERROR -> IDLE)

Live chunking (config [streaming]): while RECORDING, a segmenter thread
watches the recorder for natural pauses and drains the buffer into partial
chunks that are transcribed IMMEDIATELY — long dictations type out
progressively instead of landing in one block at the end. Partial text is
injected live only when `can_inject_now()` says it's safe (e.g. no held
hotkey modifiers); otherwise it accumulates and flushes with the final chunk,
so hold-to-talk behaves exactly like before but with the transcription
latency already paid down.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable

import numpy as np

from whisperflow.audio import SAMPLE_RATE, Recording
from whisperflow.config import StreamingConfig
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


def should_chunk(pending_s: float, since_voice_s: float, voiced: bool, st: StreamingConfig) -> bool:
    """Pure chunk-boundary decision for the segmenter loop.

    Cut when the user paused (>= pause_s of trailing silence) after having
    said something, with at least min_chunk_s buffered — or unconditionally
    at max_chunk_s so a continuous talker still streams out.
    """
    if pending_s >= st.max_chunk_s:
        return True
    return voiced and pending_s >= st.min_chunk_s and since_voice_s >= st.pause_s


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
class _Session:
    """One dictation from RECORD_START to the final inject — accumulates the
    partial chunks so history still gets a single combined entry."""

    raw_parts: list[str] = field(default_factory=list)
    injected_parts: list[str] = field(default_factory=list)
    pending_text: str = ""  # transcribed but not yet injected (modifiers held)
    context_tail: str = ""  # rolling tail of raw text, fed to later chunks as prompt
    carry: np.ndarray | None = None  # audio from a failed partial, retried on the next chunk
    cancelled: bool = False
    transcribe_failures: int = 0  # consecutive — the segmenter backs off at 2
    duration_s: float = 0.0
    transcribe_seconds: float = 0.0
    language: str = ""
    method: str = ""
    tier: str = "off"


def _append_text(base: str, piece: str) -> str:
    if not base:
        return piece
    if not piece:
        return base
    return base + " " + piece


@dataclass
class Controller:
    """Coordinates recorder, stt engine, text processor, and injector.

    recorder      : object with start() -> str, stop() -> Recording, cancel();
                    live chunking additionally needs the full _CHUNK_SURFACE
                    (take_pending/build_recording/pending_seconds/
                    seconds_since_voice/voiced_since_drain)
    engine        : SttEngine (already loaded)
    process_text  : (raw_text, language) -> (final_text, tier) — cleanup +
                    dictionary hook; identity by default (filled in step 7)
    inject_text   : (text) -> method string
    on_state      : (State, detail: str) -> None — UI callback
    on_result     : (DictationResult) -> None — history/log callback
    language      : forced language code or "" for auto
    initial_prompt: vocabulary bias string
    streaming     : StreamingConfig or None — live chunking; needs recorder.drain()
    can_inject_now: () -> bool — gate for LIVE partial injection (e.g. False
                    while hotkey modifiers are physically held); the final
                    inject always happens regardless
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
    streaming: StreamingConfig | None = None
    can_inject_now: Callable[[], bool] = field(default=lambda: True)

    _state: State = State.IDLE
    _lock: threading.Lock = field(default_factory=threading.Lock)
    _jobs: "queue.Queue" = field(default_factory=queue.Queue)
    _worker: threading.Thread | None = None
    _stopping: bool = False
    _session: _Session | None = None

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
            session, payload, is_final = job
            if session.cancelled:
                continue  # stale chunk of a cancelled dictation
            if not is_final:
                # partial jobs run while state is RECORDING and must NEVER
                # reach the ERROR path below (RECORDING -> ERROR is illegal
                # and the fallback IDLE would desync the UI from a live mic)
                self._process_partial(session, payload)
                continue
            try:
                self._process_final(session, payload)
            except Exception as exc:  # noqa: BLE001
                log.exception("dictation pipeline failed")
                self._set_state(State.ERROR, str(exc))
                self._set_state(State.IDLE)

    # ---- transcription helpers ----

    def _with_carry(self, session: _Session, recording: Recording) -> Recording:
        """Prepend audio carried over from a failed partial chunk (if any)."""
        if session.carry is None or not len(session.carry):
            return recording
        samples = np.concatenate([session.carry, recording.samples])
        session.carry = None
        return Recording(
            samples=samples,
            device_name=recording.device_name,
            duration_s=len(samples) / SAMPLE_RATE,
            rms=recording.rms,
            too_short=False,  # the carried chunk was voiced and long enough
            silent=False,
        )

    def _transcribe_piece(self, session: _Session, recording: Recording) -> str:
        """Transcribe one chunk into cleaned text ('' when skippable).

        Later chunks see a rolling tail of the session's raw text as extra
        prompt context — punctuation/casing carry across chunk boundaries.
        min_seconds is a per-DICTATION guard, not per-chunk: a tiny final
        tail after earlier chunks (a quick closing word) is still transcribed.
        """
        if recording.silent or (recording.too_short and not session.raw_parts):
            log.info(
                "skipping chunk: %s", "no speech detected" if recording.silent else "too short"
            )
            return ""
        prompt = _append_text(self.initial_prompt, session.context_tail)
        result = self.engine.transcribe(
            recording.samples,
            language=self.language,
            initial_prompt=prompt,
        )
        raw_text = result.text
        session.duration_s += result.duration_s
        session.transcribe_seconds += result.transcribe_seconds
        if result.language:
            session.language = result.language
        if not raw_text:
            return ""
        from whisperflow.stt.faster_whisper_engine import HINGLISH_SEED

        # check the echo against everything that was actually SENT as prompt —
        # whisper parrots the context tail just as happily as the vocabulary
        if is_prompt_echo(raw_text, f"{prompt} {HINGLISH_SEED}"):
            log.info("dropped prompt-echo hallucination: %r", raw_text)
            return ""

        final_text, tier = self.process_text(raw_text, result.language)
        session.raw_parts.append(raw_text)
        session.context_tail = _append_text(session.context_tail, raw_text)[-200:]
        session.tier = tier
        return final_text

    # ---- job processing ----

    def _flush_live(self, session: _Session) -> bool:
        """Inject session.pending_text mid-recording. True if it was typed.

        A clipboard fallback from the guarded injector is NOT delivery for a
        mid-dictation chunk (the next chunk would overwrite the clipboard),
        so the text stays pending and rides along to the final flush.
        """
        text = session.pending_text
        payload = (" " if session.injected_parts else "") + text
        try:
            method = self.inject_text(payload)
        except Exception:  # noqa: BLE001 — keep it pending, final flush retries
            log.exception("live injection failed — text kept for the final flush")
            return False
        if "clipboard" in method:
            log.warning("live chunk fell back to clipboard — text kept for the final flush")
            return False
        session.method = method
        session.injected_parts.append(text)
        session.pending_text = ""
        log.info("live chunk injected (%d chars) via %s", len(text), method)
        return True

    def _process_partial(self, session: _Session, payload) -> None:
        """A live chunk handed over mid-recording. Never changes state (the
        pill stays in its recording look); never raises (a lost partial must
        not kill the dictation — its audio is carried into the next chunk,
        and any exception here would corrupt the RECORDING state)."""
        try:
            recording = self._with_carry(session, self.recorder.build_recording(payload))
        except Exception:  # noqa: BLE001
            log.exception("chunk build failed — dropped")
            return
        try:
            piece = self._transcribe_piece(session, recording)
            session.transcribe_failures = 0
        except Exception:  # noqa: BLE001
            session.transcribe_failures += 1
            log.exception(
                "partial chunk failed (%d in a row) — audio carried to the next chunk",
                session.transcribe_failures,
            )
            session.carry = recording.samples
            return
        if piece:
            session.pending_text = _append_text(session.pending_text, piece)
        if not session.pending_text or session.cancelled:
            return  # Esc may have raced the transcription — do NOT type
        try:
            safe = self.can_inject_now()
        except Exception:  # noqa: BLE001
            log.exception("can_inject_now failed — deferring text to the final flush")
            safe = False
        if not safe:
            log.debug("partial held back (unsafe to type now): %r", session.pending_text)
            return
        self._flush_live(session)

    def _process_final(self, session: _Session, recording: Recording) -> None:
        recording = self._with_carry(session, recording)
        # pending/injected text implies raw_parts, so raw_parts alone tells us
        # whether the session heard anything before this final tail
        if (recording.too_short or recording.silent) and not session.raw_parts:
            reason = "too short" if recording.too_short else "no speech detected"
            log.info("skipping transcription: %s (device: %s)", reason, recording.device_name)
            self._set_state(State.IDLE, reason)
            return

        piece, error = "", None
        try:
            piece = self._transcribe_piece(session, recording)
        except Exception as exc:  # noqa: BLE001
            # the final fragment is lost, but text already transcribed or
            # injected earlier in the session MUST still be delivered/recorded
            if not session.raw_parts:
                raise
            log.exception("final chunk transcription failed — flushing what we have")
            error = exc

        flush = _append_text(session.pending_text, piece)
        session.pending_text = ""

        if not flush and not session.injected_parts:
            self._set_state(State.IDLE, "no speech detected" if not session.raw_parts else "empty transcript")
            return

        if flush:
            self._set_state(State.INJECTING)
            payload = (" " if session.injected_parts else "") + flush
            try:
                session.method = self.inject_text(payload)
                session.injected_parts.append(flush)
            except Exception as exc:  # noqa: BLE001
                if not session.injected_parts:
                    raise  # nothing was ever delivered — normal ERROR path
                log.exception("final injection failed — recording what WAS delivered")
                error = error or exc

        outcome = DictationResult(
            raw_text=" ".join(session.raw_parts),
            injected_text=" ".join(session.injected_parts),
            method=session.method or "type",
            language=session.language,
            duration_s=session.duration_s,
            transcribe_seconds=session.transcribe_seconds,
            cleanup_tier=session.tier,
        )
        try:
            self.on_result(outcome)
        except Exception:
            log.exception("on_result callback failed")
        if error is not None:
            # part of the dictation landed, part was lost — say so
            self._set_state(State.ERROR, str(error))
            self._set_state(State.IDLE)
        else:
            self._set_state(State.IDLE, f"injected via {outcome.method}")

    # ---- live chunking segmenter ----

    # the full recorder surface live chunking needs — checked as ONE contract
    # so a recorder that half-implements it can never spawn a segmenter that
    # silently dies on its first poll
    _CHUNK_SURFACE = (
        "take_pending", "build_recording", "pending_seconds",
        "seconds_since_voice", "voiced_since_drain",
    )

    def _chunking_active(self) -> bool:
        return (
            self.streaming is not None
            and self.streaming.enabled
            and all(hasattr(self.recorder, attr) for attr in self._CHUNK_SURFACE)
        )

    def _session_recording(self, session: _Session) -> bool:
        return (
            not session.cancelled
            and session is self._session
            and self._state is State.RECORDING
        )

    def _segment_loop(self, session: _Session) -> None:
        """Poll the recorder while this session records; cut chunks at pauses."""
        st = self.streaming
        while not self._stopping:
            time.sleep(0.1)
            if not self._session_recording(session):
                return
            if session.transcribe_failures >= 2:
                # engine is down — stop hammering it; audio keeps buffering
                # and the final chunk (with carried audio) surfaces the error
                log.warning("live chunking paused after repeated transcription failures")
                return
            if not should_chunk(
                self.recorder.pending_seconds,
                self.recorder.seconds_since_voice(),
                self.recorder.voiced_since_drain,
                st,
            ):
                continue
            # hand off under the controller lock so a concurrent RECORD_STOP
            # can't enqueue the final chunk between our take and our put —
            # chunk order in the worker queue must match capture order.
            # take_pending() is a cheap buffer swap; the heavy finalize work
            # happens in the worker via build_recording().
            with self._lock:
                if not self._session_recording(session):
                    return
                pending = self.recorder.take_pending()
                if pending:
                    self._jobs.put((session, pending, False))

    # ---- hotkey entry points ----

    def handle_hotkey(self, event: HotkeyEvent) -> None:
        with self._lock:
            if event is HotkeyEvent.RECORD_START:
                if self._state is not State.IDLE:
                    log.debug("RECORD_START ignored in state %s", self._state.name)
                    return
                device = self.recorder.start()
                self._session = _Session()
                if self._chunking_active():
                    threading.Thread(
                        target=self._segment_loop,
                        args=(self._session,),
                        daemon=True,
                        name="wf-segmenter",
                    ).start()
                self._set_state(State.RECORDING, device)

            elif event is HotkeyEvent.RECORD_STOP:
                if self._state is not State.RECORDING:
                    log.debug("RECORD_STOP ignored in state %s", self._state.name)
                    return
                recording = self.recorder.stop()
                self._set_state(State.TRANSCRIBING)
                self._jobs.put((self._session, recording, True))

            elif event is HotkeyEvent.RECORD_CANCEL:
                if self._state is not State.RECORDING:
                    return
                if self._session is not None:
                    self._session.cancelled = True
                self.recorder.cancel()
                self._set_state(State.IDLE, "cancelled")
