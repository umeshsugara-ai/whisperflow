# -*- coding: utf-8 -*-
"""Pill position: default math + reset (the 'I dragged it and lost it' recovery)."""

from __future__ import annotations

import tkinter as tk

import pytest

from whisperflow.ui import overlay as ov


def test_default_position_is_bottom_center():
    x, y = ov.default_position(1920, 1080, 168, 40)
    assert x == (1920 - 168) // 2
    assert y == 1080 - 40 - 80
    # never negative on any sane screen
    assert x >= 0 and y >= 0


def _tk_root():
    try:
        root = tk.Tk()
    except tk.TclError as exc:
        pytest.skip(f"no display available for a real Tk root: {exc}")
    root.withdraw()
    return root


def test_reset_position_deletes_saved_pos_and_recenters(tmp_path, monkeypatch):
    monkeypatch.setattr(ov, "POS_FILE", tmp_path / "overlay_pos.txt")
    ov.POS_FILE.write_text("9999,9999")
    root = _tk_root()
    try:
        overlay = ov.Overlay(root)
        overlay.reset_position()
        assert not ov.POS_FILE.exists()  # dragged spot forgotten
        overlay.win.update_idletasks()
        x, y = ov.default_position(
            overlay.win.winfo_screenwidth(), overlay.win.winfo_screenheight(),
            overlay.width, overlay.height,
        )
        assert f"+{x}+{y}" in overlay.win.geometry()
    finally:
        root.destroy()


def test_reset_position_without_saved_file_does_not_raise(tmp_path, monkeypatch):
    monkeypatch.setattr(ov, "POS_FILE", tmp_path / "overlay_pos.txt")
    root = _tk_root()
    try:
        overlay = ov.Overlay(root)
        overlay.reset_position()  # no file present — must be a no-op delete
    finally:
        root.destroy()


def _default_geom(overlay):
    x, y = ov.default_position(
        overlay.win.winfo_screenwidth(), overlay.win.winfo_screenheight(),
        overlay.width, overlay.height,
    )
    return f"+{x}+{y}"


def test_launch_recenters_the_pill_by_default(tmp_path, monkeypatch):
    """A pill dragged somewhere unreachable must never survive a restart —
    losing it was otherwise unrecoverable without editing files."""
    monkeypatch.setattr(ov, "POS_FILE", tmp_path / "overlay_pos.txt")
    ov.POS_FILE.write_text("5,5")  # a perfectly VALID saved spot...
    root = _tk_root()
    try:
        overlay = ov.Overlay(root)  # ...still ignored: remember_position is off
        overlay.win.update_idletasks()
        assert _default_geom(overlay) in overlay.win.geometry()
        assert ov.POS_FILE.exists()  # not deleted — opting back in restores it
    finally:
        root.destroy()


def test_remember_position_restores_a_saved_spot(tmp_path, monkeypatch):
    monkeypatch.setattr(ov, "POS_FILE", tmp_path / "overlay_pos.txt")
    ov.POS_FILE.write_text("120,300")
    root = _tk_root()
    try:
        overlay = ov.Overlay(root, remember_position=True)
        overlay.win.update_idletasks()
        assert "+120+300" in overlay.win.geometry()
    finally:
        root.destroy()


def test_remember_position_rejects_a_mostly_offscreen_spot(tmp_path, monkeypatch):
    """The old check only required the pill's top-left corner to clear the
    screen by 40px, so a pill dragged almost fully off the right edge counted
    as valid and came back just as invisible."""
    monkeypatch.setattr(ov, "POS_FILE", tmp_path / "overlay_pos.txt")
    root = _tk_root()
    try:
        probe = ov.Overlay(root, remember_position=True)
        screen_w = probe.win.winfo_screenwidth()
        # clears the old corner test, but leaves nearly the whole pill outside
        ov.POS_FILE.write_text(f"{screen_w - 45},100")
        overlay = ov.Overlay(root, remember_position=True)
        overlay.win.update_idletasks()
        assert _default_geom(overlay) in overlay.win.geometry()
    finally:
        root.destroy()


# ---- the pill can no longer be dragged somewhere it can't be grabbed back ----


def test_clamp_keeps_the_whole_pill_inside_the_area():
    area = (0, 0, 1536, 816)  # 864px screen minus a 48px taskbar
    # the exact spot the pill went missing at: y=828 is under the taskbar
    assert ov.clamp_to_area(201, 828, 168, 40, area) == (201, 776)
    # off each edge
    assert ov.clamp_to_area(-50, 400, 168, 40, area) == (0, 400)
    assert ov.clamp_to_area(2000, 400, 168, 40, area) == (1368, 400)
    assert ov.clamp_to_area(200, -30, 168, 40, area) == (200, 0)
    # already-valid spots are left exactly alone
    assert ov.clamp_to_area(300, 500, 168, 40, area) == (300, 500)


def test_clamp_respects_a_non_zero_origin():
    """A taskbar docked left/top shifts the work area's origin."""
    area = (80, 30, 1536, 864)
    assert ov.clamp_to_area(0, 0, 168, 40, area) == (80, 30)


def test_work_area_is_within_the_screen_and_never_raises():
    left, top, right, bottom = ov.work_area(1536, 864)
    assert 0 <= left < right <= 1536
    assert 0 <= top < bottom <= 864
