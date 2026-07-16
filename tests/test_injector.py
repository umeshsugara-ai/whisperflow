"""Injector tier-routing and modifier-release-wait tests (win32 monkeypatched)."""

from __future__ import annotations

import pytest

from whisperflow.config import InjectConfig
from whisperflow.inject import injector


def test_wait_returns_once_modifiers_released(monkeypatch):
    seq = iter([True, True, False])
    monkeypatch.setattr(injector, "modifiers_down", lambda: next(seq))
    sleeps = []
    monkeypatch.setattr(injector.time, "sleep", lambda s: sleeps.append(s))
    injector._wait_modifiers_released(2000)
    assert len(sleeps) == 2


def test_wait_times_out_and_proceeds(monkeypatch, caplog):
    monkeypatch.setattr(injector, "modifiers_down", lambda: True)
    clock = {"t": 0.0}
    monkeypatch.setattr(injector.time, "monotonic", lambda: clock["t"])
    monkeypatch.setattr(injector.time, "sleep", lambda s: clock.__setitem__("t", clock["t"] + 10))
    with caplog.at_level("WARNING"):
        injector._wait_modifiers_released(2000)  # must return, not hang
    assert any("still held" in r.message for r in caplog.records)


def test_wait_disabled_with_zero_timeout(monkeypatch):
    monkeypatch.setattr(
        injector, "modifiers_down", lambda: pytest.fail("must not poll when disabled")
    )
    injector._wait_modifiers_released(0)


def test_auto_routes_by_threshold_and_threads_restore_delay(monkeypatch):
    monkeypatch.setattr(injector, "_wait_modifiers_released", lambda ms: None)
    typed, pasted = [], []
    monkeypatch.setattr(
        injector.sendinput, "type_text", lambda text, interval_ms: typed.append(text)
    )
    monkeypatch.setattr(
        injector.clipboard,
        "paste_text",
        lambda text, restore_delay_ms: pasted.append((text, restore_delay_ms)),
    )
    cfg = InjectConfig(paste_threshold_chars=5, clipboard_restore_delay_ms=700)
    assert injector.inject("abc", cfg) == "type"
    assert injector.inject("abcdefgh", cfg) == "paste"
    assert typed == ["abc"]
    assert pasted == [("abcdefgh", 700)]


def test_type_failure_falls_back_to_paste(monkeypatch):
    monkeypatch.setattr(injector, "_wait_modifiers_released", lambda ms: None)
    pasted = []

    def boom(text, interval_ms):
        raise OSError("blocked")

    monkeypatch.setattr(injector.sendinput, "type_text", boom)
    monkeypatch.setattr(
        injector.clipboard,
        "paste_text",
        lambda text, restore_delay_ms: pasted.append(text),
    )
    cfg = InjectConfig(paste_threshold_chars=100)
    assert injector.inject("abc", cfg) == "paste-fallback"
    assert pasted == ["abc"]


def test_empty_text_skips_everything(monkeypatch):
    monkeypatch.setattr(
        injector, "_wait_modifiers_released", lambda ms: pytest.fail("no wait for empty text")
    )
    assert injector.inject("", InjectConfig()) == "none"
