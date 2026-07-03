"""Pure helpers behind the main window / history pane (no widgets created)."""

from __future__ import annotations

from whisperflow.ui.history_view import filter_entries
from whisperflow.ui.main_window import format_count, humanize_ts

ENTRIES = [
    {"raw": "kya haal hai", "injected": "Kya haal hai."},
    {"raw": "open the Pathlynks dashboard", "injected": "Open the Pathlynks dashboard."},
    {"raw": "hello world", "injected": "Hello world."},
]


def test_filter_entries_empty_query_returns_all():
    assert filter_entries(ENTRIES, "") == ENTRIES
    assert filter_entries(ENTRIES, "   ") == ENTRIES


def test_filter_entries_matches_raw_and_injected_case_insensitive():
    assert filter_entries(ENTRIES, "PATHLYNKS") == [ENTRIES[1]]
    assert filter_entries(ENTRIES, "haal") == [ENTRIES[0]]
    assert filter_entries(ENTRIES, "zzz") == []


def test_format_count():
    assert format_count(0) == "0"
    assert format_count(9_999) == "9,999"
    assert format_count(48_900) == "48.9K"
    assert format_count(2_500_000) == "2.5M"


def test_humanize_ts():
    assert humanize_ts("2026-07-03T14:32:10", "2026-07-03") == "Today 14:32"
    assert humanize_ts("2026-07-02T09:05:00", "2026-07-03") == "07-02 09:05"
    assert humanize_ts("", "2026-07-03") == ""
