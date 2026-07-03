"""Clipboard paste snapshot/restore semantics (win32 + SendInput monkeypatched)."""

from __future__ import annotations

from whisperflow.inject import clipboard


def _wire(monkeypatch, reads, writes):
    monkeypatch.setattr(clipboard, "_send", lambda events: None)
    monkeypatch.setattr(clipboard.time, "sleep", lambda s: None)
    it = iter(reads)
    monkeypatch.setattr(clipboard, "read_text", lambda: next(it))
    monkeypatch.setattr(clipboard, "write_text", lambda t: writes.append(t))


def test_restores_snapshot_when_clipboard_unchanged(monkeypatch):
    writes = []
    _wire(monkeypatch, ["old content", "new text"], writes)
    clipboard.paste_text("new text")
    assert writes == ["new text", "old content"]


def test_never_clobbers_third_party_clipboard_change(monkeypatch):
    writes = []
    _wire(monkeypatch, ["old content", "someone else's copy"], writes)
    clipboard.paste_text("new text")
    assert writes == ["new text"]  # snapshot NOT restored over foreign content


def test_no_restore_when_no_prior_text(monkeypatch):
    writes = []
    _wire(monkeypatch, [None], writes)  # snapshot None -> post-paste read never happens
    clipboard.paste_text("x")
    assert writes == ["x"]


def test_read_failure_skips_restore(monkeypatch):
    writes = []
    monkeypatch.setattr(clipboard, "_send", lambda events: None)
    monkeypatch.setattr(clipboard.time, "sleep", lambda s: None)
    reads = iter(["old content"])

    def read():
        try:
            return next(reads)
        except StopIteration:
            raise RuntimeError("clipboard busy")

    monkeypatch.setattr(clipboard, "read_text", read)
    monkeypatch.setattr(clipboard, "write_text", lambda t: writes.append(t))
    clipboard.paste_text("x")
    assert writes == ["x"]
