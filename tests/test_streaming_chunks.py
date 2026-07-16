"""Live-chunking tests: segmenter decision, partial injection, deferred flush.

Uses a fake recorder that exposes the chunking surface (drain/pending/voice)
with values the test controls directly — no real audio, no timing flakiness
beyond short waits on the worker/segmenter threads.
"""

from __future__ import annotations

import time

import numpy as np

from whisperflow.audio import Recording, device_warning
from whisperflow.config import StreamingConfig
from whisperflow.controller import Controller, State, should_chunk
from whisperflow.hotkey import HotkeyEvent
from whisperflow.stt.base import RawResult
from whisperflow.ui.feedback import idle_flash


def make_recording(duration=2.0, rms=0.1, too_short=False, silent=False) -> Recording:
    return Recording(
        samples=np.zeros(int(duration * 16000), dtype=np.float32),
        device_name="fake-mic",
        duration_s=duration,
        rms=rms,
        too_short=too_short,
        silent=silent,
    )


class FakeChunkRecorder:
    """Chunk-capable fake implementing the controller's full _CHUNK_SURFACE:
    the test scripts pending/voice values and the transcript each chunk
    should produce (via the paired engine)."""

    def __init__(self):
        self.pending_seconds = 0.0
        self.voiced_since_drain = False
        self._since_voice = float("inf")
        self.drained = 0
        self.final = make_recording()
        self.cancelled = False

    def seconds_since_voice(self) -> float:
        return self._since_voice

    def start(self) -> str:
        return "fake-mic"

    def stop(self) -> Recording:
        return self.final

    def cancel(self) -> None:
        self.cancelled = True

    def take_pending(self):
        self.drained += 1
        self.pending_seconds = 0.0
        self.voiced_since_drain = False
        self._since_voice = float("inf")
        return ["blocks"]  # opaque handoff token, finalized by build_recording

    def build_recording(self, pending) -> Recording:
        assert pending == ["blocks"]
        return make_recording()

    def arm_chunk(self, pending=5.0, since_voice=1.0) -> None:
        """Make the segmenter's next poll cut a chunk."""
        self.pending_seconds = pending
        self._since_voice = since_voice
        self.voiced_since_drain = True


class SequenceEngine:
    """Returns the next queued text on each transcribe call."""

    def __init__(self, texts):
        self.texts = list(texts)
        self.prompts = []

    def transcribe(self, audio, language="", initial_prompt="") -> RawResult:
        self.prompts.append(initial_prompt)
        text = self.texts.pop(0) if self.texts else ""
        return RawResult(
            text=text, language="en", language_probability=0.99,
            duration_s=2.0, transcribe_seconds=0.01,
        )


def build(recorder, engine, can_inject=lambda: True):
    states, results, injected = [], [], []

    def inject(text: str) -> str:
        injected.append(text)
        return "type"

    ctl = Controller(
        recorder=recorder,
        engine=engine,
        inject_text=inject,
        on_state=lambda s, d: states.append((s, d)),
        on_result=lambda r: results.append(r),
        streaming=StreamingConfig(enabled=True, pause_s=0.7, min_chunk_s=2.0, max_chunk_s=30.0),
        can_inject_now=can_inject,
    )
    ctl.start()
    return ctl, states, results, injected


def wait(predicate, timeout=3.0) -> None:
    deadline = time.time() + timeout
    while not predicate() and time.time() < deadline:
        time.sleep(0.02)


def wait_idle(ctl, timeout=3.0) -> None:
    wait(lambda: ctl.state is State.IDLE, timeout)


# ---- pure chunk decision ----


def test_should_chunk_needs_pause_and_min_length():
    st = StreamingConfig(pause_s=0.7, min_chunk_s=2.0, max_chunk_s=30.0)
    assert should_chunk(5.0, 1.0, True, st)
    assert not should_chunk(1.0, 1.0, True, st)  # too little buffered
    assert not should_chunk(5.0, 0.2, True, st)  # still talking
    assert not should_chunk(5.0, 1.0, False, st)  # nothing voiced yet — pure silence
    assert should_chunk(31.0, 0.0, False, st)  # force-cut at max even mid-speech


# ---- live flow ----


def test_partial_chunk_injected_live_then_final_appended():
    rec = FakeChunkRecorder()
    engine = SequenceEngine(["first sentence.", "second sentence."])
    ctl, states, results, injected = build(rec, engine)

    ctl.handle_hotkey(HotkeyEvent.RECORD_START)
    rec.arm_chunk()
    wait(lambda: len(injected) >= 1)
    assert injected == ["first sentence."]
    assert ctl.state is State.RECORDING  # pill never left the recording look

    ctl.handle_hotkey(HotkeyEvent.RECORD_STOP)
    wait_idle(ctl)
    assert injected == ["first sentence.", " second sentence."]  # leading space joins chunks
    assert len(results) == 1  # ONE combined history entry
    assert results[0].raw_text == "first sentence. second sentence."
    assert results[0].injected_text == "first sentence. second sentence."
    ctl.shutdown()


def test_partials_deferred_while_modifiers_held():
    rec = FakeChunkRecorder()
    engine = SequenceEngine(["first sentence.", "second sentence."])
    # hold-to-talk: modifiers held for the whole recording
    ctl, states, results, injected = build(rec, engine, can_inject=lambda: False)

    ctl.handle_hotkey(HotkeyEvent.RECORD_START)
    rec.arm_chunk()
    wait(lambda: len(engine.prompts) >= 1)  # first chunk transcribed in background...
    time.sleep(0.1)
    assert injected == []  # ...but nothing typed under held modifiers

    ctl.handle_hotkey(HotkeyEvent.RECORD_STOP)
    wait_idle(ctl)
    assert injected == ["first sentence. second sentence."]  # single flush at the end
    assert len(results) == 1
    ctl.shutdown()


def test_later_chunks_see_earlier_text_as_prompt_context():
    rec = FakeChunkRecorder()
    engine = SequenceEngine(["first sentence.", "second sentence."])
    ctl, states, results, injected = build(rec, engine)
    ctl.initial_prompt = "Vocab"

    ctl.handle_hotkey(HotkeyEvent.RECORD_START)
    rec.arm_chunk()
    wait(lambda: len(injected) >= 1)
    ctl.handle_hotkey(HotkeyEvent.RECORD_STOP)
    wait_idle(ctl)
    assert engine.prompts[0] == "Vocab"
    assert "first sentence." in engine.prompts[1]
    ctl.shutdown()


def test_cancel_drops_pending_partials():
    rec = FakeChunkRecorder()
    engine = SequenceEngine(["first sentence."])
    ctl, states, results, injected = build(rec, engine, can_inject=lambda: False)

    ctl.handle_hotkey(HotkeyEvent.RECORD_START)
    rec.arm_chunk()
    wait(lambda: len(engine.prompts) >= 1)
    ctl.handle_hotkey(HotkeyEvent.RECORD_CANCEL)
    time.sleep(0.2)
    assert injected == []
    assert results == []
    assert rec.cancelled
    assert ctl.state is State.IDLE
    ctl.shutdown()


def test_failed_partial_audio_carried_into_final_chunk():
    class FlakyEngine(SequenceEngine):
        def __init__(self, texts):
            super().__init__(texts)
            self.calls = 0
            self.sizes = []

        def transcribe(self, audio, language="", initial_prompt=""):
            self.calls += 1
            self.sizes.append(len(audio))
            if self.calls == 1:
                raise RuntimeError("cloud hiccup")
            return super().transcribe(audio, language, initial_prompt)

    rec = FakeChunkRecorder()
    engine = FlakyEngine(["recovered text."])
    ctl, states, results, injected = build(rec, engine)

    ctl.handle_hotkey(HotkeyEvent.RECORD_START)
    rec.arm_chunk()
    wait(lambda: engine.calls >= 1)
    ctl.handle_hotkey(HotkeyEvent.RECORD_STOP)
    wait_idle(ctl)
    # the failed chunk's audio was PREPENDED to the final chunk, not lost
    assert engine.sizes[1] == engine.sizes[0] + len(rec.final.samples)
    assert injected == ["recovered text."]
    assert ctl.state is State.IDLE
    ctl.shutdown()


def test_streaming_disabled_keeps_single_shot_behavior():
    rec = FakeChunkRecorder()
    engine = SequenceEngine(["only sentence."])
    ctl, states, results, injected = build(rec, engine)
    ctl.streaming = StreamingConfig(enabled=False)

    ctl.handle_hotkey(HotkeyEvent.RECORD_START)
    rec.arm_chunk()  # would trigger a chunk if the segmenter were running
    time.sleep(0.3)
    assert rec.drained == 0
    ctl.handle_hotkey(HotkeyEvent.RECORD_STOP)
    wait_idle(ctl)
    assert injected == ["only sentence."]
    ctl.shutdown()


def test_all_silent_session_reports_no_speech():
    rec = FakeChunkRecorder()
    rec.final = make_recording(silent=True)
    engine = SequenceEngine([])
    ctl, states, results, injected = build(rec, engine)

    ctl.handle_hotkey(HotkeyEvent.RECORD_START)
    ctl.handle_hotkey(HotkeyEvent.RECORD_STOP)
    wait_idle(ctl)
    assert injected == []
    idle_details = [d for s, d in states if s is State.IDLE]
    assert any("no speech" in d for d in idle_details)
    ctl.shutdown()


def test_partial_clipboard_fallback_defers_text():
    """A live chunk that falls back to the clipboard is NOT delivered — the
    text must stay pending and ride along to the final flush (a later chunk
    would overwrite the clipboard, silently losing the earlier one)."""
    rec = FakeChunkRecorder()
    engine = SequenceEngine(["first sentence.", "second sentence."])
    methods = iter(["clipboard (focus changed)", "type"])
    injected = []
    results = []

    def inject(text: str) -> str:
        injected.append(text)
        return next(methods)

    ctl = Controller(
        recorder=rec, engine=engine, inject_text=inject,
        on_result=lambda r: results.append(r),
        streaming=StreamingConfig(enabled=True),
        can_inject_now=lambda: True,
    )
    ctl.start()
    ctl.handle_hotkey(HotkeyEvent.RECORD_START)
    rec.arm_chunk()
    wait(lambda: len(injected) >= 1)
    ctl.handle_hotkey(HotkeyEvent.RECORD_STOP)
    wait_idle(ctl)
    # first attempt went to clipboard -> final flush retries the FULL text
    assert injected == ["first sentence.", "first sentence. second sentence."]
    assert results[0].injected_text == "first sentence. second sentence."
    ctl.shutdown()


def test_cancel_during_partial_transcription_never_types():
    """Esc while a chunk is mid-transcription must not type its text later."""
    rec = FakeChunkRecorder()
    started = []

    class SlowEngine(SequenceEngine):
        def transcribe(self, audio, language="", initial_prompt=""):
            started.append(True)
            time.sleep(0.3)
            return super().transcribe(audio, language, initial_prompt)

    engine = SlowEngine(["late text."])
    ctl, states, results, injected = build(rec, engine)
    ctl.handle_hotkey(HotkeyEvent.RECORD_START)
    rec.arm_chunk()
    wait(lambda: bool(started))  # transcription in flight
    ctl.handle_hotkey(HotkeyEvent.RECORD_CANCEL)
    time.sleep(0.6)  # let the slow transcription finish
    assert injected == []
    assert results == []
    ctl.shutdown()


def test_final_error_still_flushes_pending_and_records():
    """Hold-to-talk: chunk 1 transcribed but held back; the final chunk's
    transcription fails -> the held text must still be injected + recorded,
    and the error surfaced afterwards."""
    rec = FakeChunkRecorder()

    class FinalFailsEngine(SequenceEngine):
        def transcribe(self, audio, language="", initial_prompt=""):
            if not self.texts:
                raise RuntimeError("network died")
            return super().transcribe(audio, language, initial_prompt)

    engine = FinalFailsEngine(["first sentence."])
    ctl, states, results, injected = build(rec, engine, can_inject=lambda: False)
    ctl.handle_hotkey(HotkeyEvent.RECORD_START)
    rec.arm_chunk()
    wait(lambda: len(engine.prompts) >= 1)
    ctl.handle_hotkey(HotkeyEvent.RECORD_STOP)
    wait_idle(ctl)
    assert injected == ["first sentence."]  # held text delivered despite the error
    assert len(results) == 1 and results[0].injected_text == "first sentence."
    assert any(s is State.ERROR for s, d in states)  # partial loss surfaced
    assert ctl.state is State.IDLE
    ctl.shutdown()


def test_tiny_final_tail_after_chunks_still_transcribed():
    """min_seconds is a per-dictation guard: a quick closing word after a
    chunk boundary must not be silently dropped as 'too short'."""
    rec = FakeChunkRecorder()
    rec.final = make_recording(duration=0.2, too_short=True)
    engine = SequenceEngine(["long sentence.", "thanks."])
    ctl, states, results, injected = build(rec, engine)
    ctl.handle_hotkey(HotkeyEvent.RECORD_START)
    rec.arm_chunk()
    wait(lambda: len(injected) >= 1)
    ctl.handle_hotkey(HotkeyEvent.RECORD_STOP)
    wait_idle(ctl)
    assert injected == ["long sentence.", " thanks."]
    ctl.shutdown()


# ---- mic feedback (pure helpers) ----


def test_device_warning_flags_fallback_and_virtual_mics():
    # pinned mic missing -> fallback warning names both devices
    w = device_warning("Realtek(R) Audio", "Microphone (Camo)")
    assert "Realtek(R) Audio" in w and "Camo" in w
    # virtual mic as default -> silence warning
    w = device_warning("default", "Microphone (Camo)")
    assert "virtual" in w.lower()
    # healthy cases -> no warning
    assert device_warning("default", "Realtek(R) Audio Microphone") == ""
    assert device_warning("Realtek", "Realtek(R) Audio Microphone") == ""


def test_list_input_devices_prefers_wasapi_and_dedupes():
    from whisperflow.audio import list_input_devices

    hostapis = [{"name": "MME"}, {"name": "Windows WASAPI"}]
    devices = [
        # MME truncates to 31 chars and lists the same mic again
        {"name": "Microphone (Realtek(R) Audio)", "max_input_channels": 2, "hostapi": 0},
        {"name": "Microphone (Realtek(R) Audio)", "max_input_channels": 2, "hostapi": 1},
        {"name": "Microphone (Camo)", "max_input_channels": 1, "hostapi": 1},
        # outputs never show up
        {"name": "Speakers (Realtek(R) Audio)", "max_input_channels": 0, "hostapi": 1},
    ]
    assert list_input_devices(devices, hostapis) == [
        "Microphone (Realtek(R) Audio)",
        "Microphone (Camo)",
    ]


def test_list_input_devices_falls_back_without_wasapi():
    from whisperflow.audio import list_input_devices

    hostapis = [{"name": "Core Audio"}]
    devices = [
        {"name": "Built-in Microphone", "max_input_channels": 1, "hostapi": 0},
        {"name": "built-in microphone", "max_input_channels": 1, "hostapi": 0},
    ]
    # no WASAPI rows -> all-API dedup (case-insensitive)
    assert list_input_devices(devices, hostapis) == ["Built-in Microphone"]


def test_idle_flash_covers_silent_and_short_outcomes():
    assert idle_flash("no speech detected") == ("warn", "No speech — check mic ⚠")
    assert idle_flash("too short") == ("warn", "Too short — hold & speak")
    assert idle_flash("empty transcript") == ("warn", "No speech — check mic ⚠")
    assert idle_flash("injected via type") == ("done", "Injected ✓")
    assert idle_flash("clipboard (focus changed)") == ("warn", "Copied — press Ctrl+V")
    assert idle_flash("") is None
    assert idle_flash("cancelled") is None
    # every message fits the pill's 28-char label
    for detail in ("no speech detected", "too short", "injected via type", "clipboard"):
        flash = idle_flash(detail)
        assert flash is None or len(flash[1]) <= 28
