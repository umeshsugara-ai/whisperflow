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


def make_recording(duration=2.0, rms=0.1, too_short=False, silent=False, low_level=False) -> Recording:
    return Recording(
        samples=np.zeros(int(duration * 16000), dtype=np.float32),
        device_name="fake-mic",
        duration_s=duration,
        rms=rms,
        too_short=too_short,
        silent=silent,
        low_level=low_level,
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
    assert idle_flash("mic unavailable") == ("warn", "Mic busy — try again ⚠")
    assert idle_flash("") is None
    assert idle_flash("cancelled") is None
    # every message fits the pill's 28-char label
    for detail in ("no speech detected", "too short", "injected via type", "clipboard", "mic unavailable"):
        flash = idle_flash(detail)
        assert flash is None or len(flash[1]) <= 28


def test_resolve_device_prefers_wasapi_row_over_directsound():
    """The Settings picker pins full WASAPI names; the resolver must pick
    that mic's WASAPI row, not its DirectSound row (first substring match) —
    DirectSound is the API that dies with PaErrorCode -9999 after
    sleep/lock, which broke a correctly-pinned mic."""
    from whisperflow.audio import resolve_device

    hostapis = [{"name": "MME"}, {"name": "Windows DirectSound"}, {"name": "Windows WASAPI"}]
    devices = [
        # MME's 31-char truncation: needle "realtek(r) audio" doesn't match
        {"name": "Microphone Array (Realtek(R) Au", "max_input_channels": 2, "hostapi": 0},
        {"name": "Microphone Array (Realtek(R) Audio)", "max_input_channels": 2, "hostapi": 1},
        {"name": "Microphone Array (Realtek(R) Audio)", "max_input_channels": 2, "hostapi": 2},
    ]
    idx, name = resolve_device("Realtek(R) Audio", devices, hostapis)
    assert idx == 2  # the WASAPI row, not the first (DirectSound) match
    assert name == "Microphone Array (Realtek(R) Audio)"


def test_resolve_device_falls_back_to_first_match_without_wasapi():
    from whisperflow.audio import resolve_device

    hostapis = [{"name": "Core Audio"}]
    devices = [
        {"name": "Built-in Microphone", "max_input_channels": 1, "hostapi": 0},
    ]
    idx, name = resolve_device("built-in", devices, hostapis)
    assert idx == 0
    assert name == "Built-in Microphone"


def test_uses_wasapi_true_only_for_wasapi_hosted_devices():
    """WASAPI streams need auto_convert (PortAudio's WASAPI backend rejects
    our 16 kHz with -9997 'Invalid sample rate' when the device mix format
    is 48 kHz) — but WasapiSettings must NOT be passed to MME/DirectSound
    streams, so the decision has to name the right host API per index."""
    from whisperflow.audio import uses_wasapi

    hostapis = [{"name": "MME"}, {"name": "Windows DirectSound"}, {"name": "Windows WASAPI"}]
    devices = [
        {"name": "Mic (Realtek)", "max_input_channels": 2, "hostapi": 0},
        {"name": "Mic (Realtek)", "max_input_channels": 2, "hostapi": 1},
        {"name": "Mic (Realtek)", "max_input_channels": 2, "hostapi": 2},
    ]
    assert uses_wasapi(2, devices, hostapis) is True
    assert uses_wasapi(0, devices, hostapis) is False
    assert uses_wasapi(1, devices, hostapis) is False


def test_uses_wasapi_never_raises_on_bad_input():
    from whisperflow.audio import uses_wasapi

    hostapis = [{"name": "Windows WASAPI"}]
    devices = [{"name": "Mic", "max_input_channels": 1, "hostapi": 0}]
    assert uses_wasapi(99, devices, hostapis) is False  # index out of range
    assert uses_wasapi(-1, devices, hostapis) is False


# ---- capped-gain silence guard (_finalize) ----
# Live incident 2026-07-28: mic level collapsed ~20x; noise-floor chunks
# (rms 0.0005-0.0006, gain pinned at the 40x cap) passed the plain silence
# gate and whisper hallucinated 207 chars + a raw "<|hi|>" token into the
# user's window. These tests pin the band logic that closes that hole.

def _finalize_const(amplitude: float, seconds: float = 1.0):
    """Finalize one constant-amplitude block: rms == peak == amplitude."""
    from whisperflow.audio import Recorder
    from whisperflow.config import AudioConfig

    rec = Recorder(AudioConfig())  # no stream is opened by __init__
    block = np.full(int(seconds * 16000), amplitude, dtype=np.float32)
    return rec._finalize([block])


def test_capped_gain_band_is_silent_and_low_level():
    # THE regression: 0.0006 was silent=False before the guard (0.0006 > 0.0005)
    r = _finalize_const(0.0006)
    assert r.silent is True
    assert r.low_level is True


def test_plain_silence_stays_low_level_false():
    r = _finalize_const(0.0004)
    assert r.silent is True
    assert r.low_level is False  # ordinary quiet room -> ordinary "No speech" UX


def test_quiet_real_speech_still_transcribes_with_gain():
    r = _finalize_const(0.006)
    assert r.silent is False
    assert r.low_level is False
    assert float(np.abs(r.samples).max()) > 0.006  # gain was applied


def test_healthy_and_loud_audio_unchanged():
    healthy = _finalize_const(0.05)
    assert healthy.silent is False and healthy.low_level is False
    loud = _finalize_const(0.4)
    assert loud.silent is False
    assert float(np.abs(loud.samples).max()) == np.float32(0.4)  # >=0.30: no gain


def test_empty_blocks_finalize_silent_without_crash():
    from whisperflow.audio import Recorder
    from whisperflow.config import AudioConfig

    r = Recorder(AudioConfig())._finalize([])
    assert r.silent is True and r.low_level is False and r.duration_s == 0.0


def test_silent_recordings_are_never_amplified(caplog):
    import logging

    with caplog.at_level(logging.INFO, logger="whisperflow.audio"):
        r = _finalize_const(0.0006)
    assert float(np.abs(r.samples).max()) == np.float32(0.0006)  # untouched
    assert not any("auto-gain applied" in m for m in caplog.messages)


def test_low_level_streak_warns_once_and_rearms(caplog):
    import logging

    from whisperflow.audio import Recorder
    from whisperflow.config import AudioConfig

    rec = Recorder(AudioConfig())
    in_band = [np.full(16000, 0.0006, dtype=np.float32)]
    plain_silent = [np.full(16000, 0.0004, dtype=np.float32)]
    healthy = [np.full(16000, 0.05, dtype=np.float32)]

    def warnings():
        return [m for m in caplog.messages if "Mic level near zero" in m]

    with caplog.at_level(logging.WARNING, logger="whisperflow.audio"):
        rec._finalize(in_band)
        rec._finalize(plain_silent)  # neutral: must NOT reset the streak
        rec._finalize(in_band)
        assert warnings() == []
        rec._finalize(in_band)  # third in-band -> the one warning
        assert len(warnings()) == 1
        rec._finalize(in_band)  # fourth -> no spam
        assert len(warnings()) == 1
        rec._finalize(healthy)  # real speech resets + re-arms
        for _ in range(3):
            rec._finalize(in_band)
        assert len(warnings()) == 2


def test_idle_flash_maps_mic_level_detail():
    flash = idle_flash("mic level near zero")
    assert flash == ("warn", "Mic level near zero ⚠")
    assert len(flash[1]) <= 28
    # existing mappings untouched
    assert idle_flash("no speech detected") == ("warn", "No speech — check mic ⚠")
    assert idle_flash("mic unavailable") == ("warn", "Mic busy — try again ⚠")


def test_low_level_final_reports_mic_level_not_no_speech():
    rec = FakeChunkRecorder()
    rec.final = make_recording(silent=True, low_level=True)
    ctl, states, results, injected = build(rec, SequenceEngine([]))

    ctl.handle_hotkey(HotkeyEvent.RECORD_START)
    ctl.handle_hotkey(HotkeyEvent.RECORD_STOP)
    wait_idle(ctl)
    idle_details = [d for s, d in states if s is State.IDLE]
    assert any("mic level near zero" in d for d in idle_details)
    assert not any("no speech" in d for d in idle_details)
    ctl.shutdown()


def test_too_short_wins_over_low_level():
    rec = FakeChunkRecorder()
    rec.final = make_recording(duration=0.1, too_short=True, silent=True, low_level=True)
    ctl, states, results, injected = build(rec, SequenceEngine([]))

    ctl.handle_hotkey(HotkeyEvent.RECORD_START)
    ctl.handle_hotkey(HotkeyEvent.RECORD_STOP)
    wait_idle(ctl)
    idle_details = [d for s, d in states if s is State.IDLE]
    assert any("too short" in d for d in idle_details)
    ctl.shutdown()


# ---- whisper special-token scrub ----

def test_special_tokens_scrubbed_from_transcripts():
    rec = FakeChunkRecorder()
    engine = SequenceEngine(["<|hi|> chalo shuru karte hain"])
    ctl, states, results, injected = build(rec, engine)

    ctl.handle_hotkey(HotkeyEvent.RECORD_START)
    ctl.handle_hotkey(HotkeyEvent.RECORD_STOP)
    wait_idle(ctl)
    assert injected == ["chalo shuru karte hain"]  # token gone, text intact
    ctl.shutdown()


def test_token_only_transcript_treated_as_empty():
    rec = FakeChunkRecorder()
    engine = SequenceEngine(["<|hi|><|endoftext|>"])
    ctl, states, results, injected = build(rec, engine)

    ctl.handle_hotkey(HotkeyEvent.RECORD_START)
    ctl.handle_hotkey(HotkeyEvent.RECORD_STOP)
    wait_idle(ctl)
    assert injected == []
    assert results == []  # nothing delivered, nothing recorded
    ctl.shutdown()


def test_scrub_leaves_ordinary_angle_brackets_alone():
    from whisperflow.controller import _SPECIAL_TOKEN_RE

    assert _SPECIAL_TOKEN_RE.sub("", "a < b | c > d") == "a < b | c > d"
    assert _SPECIAL_TOKEN_RE.sub("", "x <|nospeech|> y") == "x  y"
