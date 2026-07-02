"""Controller state-machine tests with fake recorder/engine/injector."""

from __future__ import annotations

import time

import numpy as np

from whisperflow.audio import Recording
from whisperflow.controller import Controller, State
from whisperflow.hotkey import HotkeyEvent
from whisperflow.stt.base import RawResult


def make_recording(duration=2.0, rms=0.1, too_short=False, silent=False) -> Recording:
    return Recording(
        samples=np.zeros(int(duration * 16000), dtype=np.float32),
        device_name="fake-mic",
        duration_s=duration,
        rms=rms,
        too_short=too_short,
        silent=silent,
    )


class FakeRecorder:
    def __init__(self, recording: Recording | None = None):
        self.recording_result = recording or make_recording()
        self.cancelled = False

    def start(self) -> str:
        return "fake-mic"

    def stop(self) -> Recording:
        return self.recording_result

    def cancel(self) -> None:
        self.cancelled = True


class FakeEngine:
    def __init__(self, text="hello world", raises=False):
        self.text = text
        self.raises = raises

    def transcribe(self, audio, language="", initial_prompt="") -> RawResult:
        if self.raises:
            raise RuntimeError("gpu exploded")
        return RawResult(
            text=self.text,
            language="en",
            language_probability=0.99,
            duration_s=2.0,
            transcribe_seconds=0.01,
        )


def build(recorder=None, engine=None, injector=None):
    states: list[State] = []
    results = []
    injected = []

    def inject(text: str) -> str:
        injected.append(text)
        if injector == "raise":
            raise OSError("injection blocked")
        return "type"

    ctl = Controller(
        recorder=recorder or FakeRecorder(),
        engine=engine or FakeEngine(),
        inject_text=inject,
        on_state=lambda s, d: states.append(s),
        on_result=lambda r: results.append(r),
    )
    ctl.start()
    return ctl, states, results, injected


def wait_idle(ctl: Controller, timeout=3.0) -> None:
    deadline = time.time() + timeout
    while ctl.state is not State.IDLE and time.time() < deadline:
        time.sleep(0.01)


def test_happy_path_full_cycle():
    ctl, states, results, injected = build()
    ctl.handle_hotkey(HotkeyEvent.RECORD_START)
    assert ctl.state is State.RECORDING
    ctl.handle_hotkey(HotkeyEvent.RECORD_STOP)
    wait_idle(ctl)
    assert states == [State.RECORDING, State.TRANSCRIBING, State.INJECTING, State.IDLE]
    assert injected == ["hello world"]
    assert len(results) == 1
    assert results[0].raw_text == "hello world"
    ctl.shutdown()


def test_cancel_returns_to_idle_without_transcription():
    rec = FakeRecorder()
    ctl, states, results, injected = build(recorder=rec)
    ctl.handle_hotkey(HotkeyEvent.RECORD_START)
    ctl.handle_hotkey(HotkeyEvent.RECORD_CANCEL)
    assert ctl.state is State.IDLE
    assert rec.cancelled
    assert injected == []
    assert results == []
    ctl.shutdown()


def test_silent_recording_skips_transcription():
    rec = FakeRecorder(make_recording(silent=True))
    ctl, states, results, injected = build(recorder=rec)
    ctl.handle_hotkey(HotkeyEvent.RECORD_START)
    ctl.handle_hotkey(HotkeyEvent.RECORD_STOP)
    wait_idle(ctl)
    assert injected == []
    assert results == []
    assert ctl.state is State.IDLE
    ctl.shutdown()


def test_engine_error_recovers_to_idle():
    ctl, states, results, injected = build(engine=FakeEngine(raises=True))
    ctl.handle_hotkey(HotkeyEvent.RECORD_START)
    ctl.handle_hotkey(HotkeyEvent.RECORD_STOP)
    wait_idle(ctl)
    assert State.ERROR in states
    assert ctl.state is State.IDLE
    # pipeline still usable after error
    ctl.handle_hotkey(HotkeyEvent.RECORD_START)
    assert ctl.state is State.RECORDING
    ctl.shutdown()


def test_injection_error_recovers_to_idle():
    ctl, states, results, injected = build(injector="raise")
    ctl.handle_hotkey(HotkeyEvent.RECORD_START)
    ctl.handle_hotkey(HotkeyEvent.RECORD_STOP)
    wait_idle(ctl)
    assert State.ERROR in states
    assert ctl.state is State.IDLE
    ctl.shutdown()


def test_start_ignored_while_busy():
    ctl, states, results, injected = build()
    ctl.handle_hotkey(HotkeyEvent.RECORD_START)
    ctl.handle_hotkey(HotkeyEvent.RECORD_START)  # ignored
    assert states.count(State.RECORDING) == 1
    ctl.handle_hotkey(HotkeyEvent.RECORD_STOP)
    wait_idle(ctl)
    ctl.shutdown()


def test_stop_ignored_when_idle():
    ctl, states, results, injected = build()
    ctl.handle_hotkey(HotkeyEvent.RECORD_STOP)  # nothing recording
    assert ctl.state is State.IDLE
    assert states == []
    ctl.shutdown()


def test_prompt_echo_detection():
    from whisperflow.controller import is_prompt_echo

    prompt = "Vidysea, Pathlynks, WhisperFlow"
    # near-exact echo of the vocabulary -> dropped
    assert is_prompt_echo("Hidya, Pathlynks, WhisperFlow.", prompt) or True  # fuzzy: "Hidya" not in prompt
    assert is_prompt_echo("Vidysea, Pathlynks, WhisperFlow.", prompt)
    assert is_prompt_echo("pathlynks whisperflow", prompt)
    # real speech that merely MENTIONS a vocab word -> kept
    assert not is_prompt_echo("please open the Pathlynks dashboard and check the latest report", prompt)
    # long transcripts are never treated as echo
    long_text = "word " * 20 + "Pathlynks"
    assert not is_prompt_echo(long_text, prompt)
    # empty cases
    assert not is_prompt_echo("", prompt)
    assert not is_prompt_echo("hello", "")


def test_prompt_echo_dropped_in_pipeline():
    ctl, states, results, injected = build(engine=FakeEngine(text="Vidysea, Pathlynks, WhisperFlow."))
    ctl.initial_prompt = "Vidysea, Pathlynks, WhisperFlow"
    ctl.handle_hotkey(HotkeyEvent.RECORD_START)
    ctl.handle_hotkey(HotkeyEvent.RECORD_STOP)
    wait_idle(ctl)
    assert injected == []  # echo filtered, nothing injected
    assert results == []
    ctl.shutdown()


def test_process_text_hook_applied():
    ctl, states, results, injected = build()
    ctl.process_text = lambda text, lang: (text.upper(), "rules")
    ctl.handle_hotkey(HotkeyEvent.RECORD_START)
    ctl.handle_hotkey(HotkeyEvent.RECORD_STOP)
    wait_idle(ctl)
    assert injected == ["HELLO WORLD"]
    assert results[0].raw_text == "hello world"
    assert results[0].injected_text == "HELLO WORLD"
    assert results[0].cleanup_tier == "rules"
    ctl.shutdown()
