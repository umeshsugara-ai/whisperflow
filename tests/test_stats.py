"""Lifetime-stats rollup: pure functions + History.stats() persistence."""

from __future__ import annotations

import json

from whisperflow.history import (
    History,
    accumulate,
    average_wpm,
    compute_streak,
    count_words,
    empty_stats,
)


def entry(ts="2026-07-03T10:00:00", injected="hello world", duration_s=3.0) -> dict:
    return {
        "ts": ts,
        "raw": injected,
        "injected": injected,
        "tier": "rules",
        "method": "type",
        "language": "en",
        "duration_s": duration_s,
        "latency_ms": 500.0,
    }


def test_count_words():
    assert count_words("") == 0
    assert count_words("hello world") == 2
    assert count_words("  kya tum   sun rahe ho ") == 5  # unicode + odd spacing


def test_accumulate_folds_words_days_and_duration():
    s = empty_stats()
    accumulate(s, entry(ts="2026-07-01T09:00:00", injected="a b c", duration_s=6.0))
    accumulate(s, entry(ts="2026-07-01T10:00:00", injected="d e", duration_s=4.0))
    accumulate(s, entry(ts="2026-07-02T10:00:00", injected="f", duration_s=2.0))
    assert s["total_words"] == 6
    assert s["total_dictations"] == 3
    assert s["total_speaking_s"] == 12.0
    assert s["days"] == {"2026-07-01": 2, "2026-07-02": 1}


def test_compute_streak():
    days = {"2026-07-01": 1, "2026-07-02": 3, "2026-07-03": 2}
    assert compute_streak(days, "2026-07-03") == 3
    assert compute_streak(days, "2026-07-04") == 3  # streak ending yesterday still counts
    assert compute_streak(days, "2026-07-05") == 0  # gap breaks it
    assert compute_streak({}, "2026-07-03") == 0
    assert compute_streak(days, "not-a-date") == 0
    assert compute_streak({"2026-07-01": 1, "2026-07-03": 1}, "2026-07-03") == 1  # gap on 07-02


def test_average_wpm():
    assert average_wpm(empty_stats()) == 0.0  # zero-duration guard
    s = {"total_words": 120, "total_speaking_s": 60.0}
    assert average_wpm(s) == 120.0


def test_lifetime_totals_survive_trim(tmp_path):
    h = History(tmp_path / "history.jsonl", max_entries=5)
    for i in range(12):
        h.append(
            raw=f"word{i} extra",
            injected=f"word{i} extra",
            tier="rules",
            method="type",
            language="en",
            duration_s=3.0,
            latency_ms=100.0,
        )
    assert len(h.entries(limit=100)) == 5  # trimmed
    s = h.stats()
    assert s["total_dictations"] == 12  # lifetime survives the trim
    assert s["total_words"] == 24
    assert s["total_speaking_s"] == 36.0


def test_stats_seed_from_legacy_jsonl(tmp_path):
    path = tmp_path / "history.jsonl"
    lines = [json.dumps(entry(injected="one two three")) for _ in range(3)]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    h = History(path)  # no stats.json yet — legacy install
    s = h.stats()
    assert s["total_dictations"] == 3
    assert s["total_words"] == 9
    assert h.stats_path.exists()


def test_corrupt_stats_json_reseeds_without_crash(tmp_path):
    path = tmp_path / "history.jsonl"
    h = History(path)
    h.append(
        raw="a b", injected="a b", tier="rules", method="type",
        language="en", duration_s=2.0, latency_ms=10.0,
    )
    h.stats_path.write_text("{not json", encoding="utf-8")
    s = h.stats()
    assert s["total_dictations"] == 1
    assert s["total_words"] == 2


def test_clear_keeps_lifetime_stats(tmp_path):
    h = History(tmp_path / "history.jsonl")
    h.append(
        raw="a b", injected="a b", tier="rules", method="type",
        language="en", duration_s=2.0, latency_ms=10.0,
    )
    h.clear()
    assert h.entries() == []
    assert h.stats()["total_dictations"] == 1
