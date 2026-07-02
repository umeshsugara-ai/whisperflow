# -*- coding: utf-8 -*-
"""History JSONL — the raw transcript must always be recoverable."""

import json

from whisperflow.history import History


def make(tmp_path, max_entries=500):
    return History(tmp_path / "history.jsonl", max_entries=max_entries)


def append_n(hist, n, **overrides):
    for i in range(n):
        hist.append(
            raw=overrides.get("raw", f"raw text {i}"),
            injected=overrides.get("injected", f"cleaned text {i}"),
            tier="rules",
            method="type",
            language="en",
            duration_s=2.0,
            latency_ms=800.0,
        )


def test_raw_and_injected_both_stored(tmp_path):
    hist = make(tmp_path)
    hist.append(
        raw="um hello world yaar",
        injected="Hello world.",
        tier="rules",
        method="type",
        language="en",
        duration_s=1.5,
        latency_ms=650.0,
    )
    entry = hist.last()
    assert entry["raw"] == "um hello world yaar"
    assert entry["injected"] == "Hello world."
    assert entry["tier"] == "rules"


def test_devanagari_stored_unescaped(tmp_path):
    hist = make(tmp_path)
    hist.append(
        raw="नमस्ते दुनिया",
        injected="नमस्ते दुनिया",
        tier="off",
        method="type",
        language="hi",
        duration_s=1.0,
        latency_ms=500.0,
    )
    text = hist.path.read_text(encoding="utf-8")
    assert "नमस्ते" in text  # ensure_ascii=False
    assert hist.last()["raw"] == "नमस्ते दुनिया"


def test_trim_to_max_entries(tmp_path):
    hist = make(tmp_path, max_entries=10)
    append_n(hist, 25)
    lines = hist.path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 10
    assert json.loads(lines[-1])["raw"] == "raw text 24"  # newest kept


def test_entries_and_clear(tmp_path):
    hist = make(tmp_path)
    append_n(hist, 3)
    assert len(hist.entries()) == 3
    hist.clear()
    assert hist.entries() == []
    assert hist.last() is None


def test_corrupt_line_skipped(tmp_path):
    hist = make(tmp_path)
    append_n(hist, 2)
    with open(hist.path, "a", encoding="utf-8") as f:
        f.write("{not json}\n")
    append_n(hist, 1)
    entries = hist.entries()
    assert len(entries) == 3  # corrupt line dropped, valid ones kept
