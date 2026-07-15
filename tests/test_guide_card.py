"""First-run "How to use" card — dismissal persistence + content."""

from __future__ import annotations

from whisperflow.ui import main_window as mw


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
