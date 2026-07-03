"""Focus-guard logic tests — win32 calls monkeypatched, no real windows."""

from __future__ import annotations

import pytest

from whisperflow.config import InjectConfig
from whisperflow.inject import focus


@pytest.fixture()
def cfg() -> InjectConfig:
    return InjectConfig()


def test_matching_focus_injects_directly(monkeypatch, cfg):
    monkeypatch.setattr(focus, "current_window", lambda: 222)
    monkeypatch.setattr(focus, "try_activate", lambda hwnd: pytest.fail("no activation needed"))
    monkeypatch.setattr(focus.injector, "inject", lambda text, c: "type")
    assert focus.inject_guarded("hi", cfg, expected_hwnd=222) == "type"


def test_no_expected_hwnd_injects_directly(monkeypatch, cfg):
    monkeypatch.setattr(focus, "current_window", lambda: pytest.fail("no guard without hwnd"))
    monkeypatch.setattr(focus.injector, "inject", lambda text, c: "type")
    assert focus.inject_guarded("hi", cfg, expected_hwnd=0) == "type"


def test_own_window_foreground_restores_and_injects(monkeypatch, cfg):
    """The pill/tray grabbing foreground must NOT dump text to the clipboard."""
    activated = []
    monkeypatch.setattr(focus, "current_window", lambda: 111)
    monkeypatch.setattr(focus, "_window_pid", lambda hwnd: 999)
    monkeypatch.setattr(focus.os, "getpid", lambda: 999)
    monkeypatch.setattr(focus, "window_title", lambda hwnd: "pill")
    monkeypatch.setattr(focus, "try_activate", lambda hwnd: activated.append(hwnd) or True)
    monkeypatch.setattr(focus.injector, "inject", lambda text, c: "type")
    assert focus.inject_guarded("hi", cfg, expected_hwnd=222) == "type"
    assert activated == [222]


def test_foreign_focus_restored_then_injects(monkeypatch, cfg):
    monkeypatch.setattr(focus, "current_window", lambda: 111)
    monkeypatch.setattr(focus, "_window_pid", lambda hwnd: 4242)
    monkeypatch.setattr(focus.os, "getpid", lambda: 999)
    monkeypatch.setattr(focus, "window_title", lambda hwnd: "other app")
    monkeypatch.setattr(focus, "try_activate", lambda hwnd: True)
    monkeypatch.setattr(focus.injector, "inject", lambda text, c: "type")
    assert focus.inject_guarded("hi", cfg, expected_hwnd=222) == "type"


def test_unrecoverable_focus_goes_to_clipboard(monkeypatch, cfg):
    written = []
    monkeypatch.setattr(focus, "current_window", lambda: 111)
    monkeypatch.setattr(focus, "_window_pid", lambda hwnd: 4242)
    monkeypatch.setattr(focus.os, "getpid", lambda: 999)
    monkeypatch.setattr(focus, "window_title", lambda hwnd: "other app")
    monkeypatch.setattr(focus, "try_activate", lambda hwnd: False)
    monkeypatch.setattr(focus.clipboard, "write_text", lambda t: written.append(t))
    monkeypatch.setattr(
        focus.injector, "inject", lambda *a: pytest.fail("must not inject into wrong window")
    )
    assert focus.inject_guarded("hi", cfg, expected_hwnd=222) == "clipboard (focus changed)"
    assert written == ["hi"]
