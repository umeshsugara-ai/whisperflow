"""First-run "How to use" card — dismissal persistence + content."""

from __future__ import annotations

import tkinter as tk

import pytest

from whisperflow.config import Config
from whisperflow.history import History
from whisperflow.ui import main_window as mw


def _tk_root():
    try:
        root = tk.Tk()
    except tk.TclError as exc:
        pytest.skip(f"no display available for a real Tk root: {exc}")
    root.withdraw()
    return root


def test_guide_not_dismissed_initially(tmp_path, monkeypatch):
    monkeypatch.setattr(mw, "data_dir", lambda: tmp_path)
    assert not mw.guide_dismissed()


def test_dismiss_guide_persists(tmp_path, monkeypatch):
    monkeypatch.setattr(mw, "data_dir", lambda: tmp_path)
    mw.dismiss_guide()
    assert mw.guide_dismissed()
    assert (tmp_path / mw.GUIDE_DISMISSED_FILE).exists()


def test_guide_lines_use_real_hotkey_label():
    lines = mw.guide_lines("Alt+Win")
    gestures = [g for g, _ in lines]
    assert "Hold Alt+Win" in gestures
    assert "Tap Alt+Win" in gestures
    assert "Esc" in gestures
    # every row explains itself in plain language
    assert all(what for _, what in lines)


def test_guide_card_gesture_labels_follow_hotkey_changed_after_first_open(tmp_path, monkeypatch):
    """MainWindow is a singleton (whisperflow/ui/main_window.py:147) so
    HomePage.__init__ — and the guide card it builds — runs only once per
    process. A hotkey changed later in Settings must still update this
    card on the next refresh(), not leave it showing the combo from
    whenever the window first opened."""
    monkeypatch.setattr(mw, "data_dir", lambda: tmp_path)
    root = _tk_root()
    try:
        cfg = Config()
        cfg.hotkey.combo = "alt+windows"
        history = History(tmp_path / "history.jsonl")
        page = mw.HomePage(root, history, warnings_source=None, open_history=lambda: None, cfg=cfg)
        assert [w.cget("text") for w in page._guide_gesture_labels] == [
            "Hold Alt+Win", "Tap Alt+Win", "Esc",
        ]

        cfg.hotkey.combo = "ctrl+windows"  # e.g. a Settings Save, without reopening the window
        page.refresh()

        assert [w.cget("text") for w in page._guide_gesture_labels] == [
            "Hold Ctrl+Win", "Tap Ctrl+Win", "Esc",
        ]
    finally:
        root.destroy()


def test_guide_card_refresh_after_dismissal_does_not_raise(tmp_path):
    """"Got it" destroys the card; a refresh() that fires afterwards (the
    next show_page("home")) must not crash on the now-dead widgets."""
    root = _tk_root()
    try:
        cfg = Config()
        history = History(tmp_path / "history.jsonl")
        page = mw.HomePage(root, history, warnings_source=None, open_history=lambda: None, cfg=cfg)
        card = page._guide_gesture_labels[0].master.master  # row -> card
        card.destroy()  # simulates the "Got it" button's card.destroy()

        page.refresh()  # must not raise
        assert page._guide_gesture_labels == []
    finally:
        root.destroy()
