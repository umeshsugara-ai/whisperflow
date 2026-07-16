"""Pure helpers behind the main window / history pane (no widgets created)."""

from __future__ import annotations

from whisperflow.ui.history_view import filter_entries
from whisperflow.ui.main_window import (
    MIC_DEFAULT_LABEL,
    format_count,
    humanize_ts,
    mic_choice_to_config,
    mic_config_to_choice,
)

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


def test_mic_choice_round_trip():
    # the follow-Windows row maps to "default" and back
    assert mic_choice_to_config(MIC_DEFAULT_LABEL) == "default"
    assert mic_config_to_choice("default") == MIC_DEFAULT_LABEL
    assert mic_config_to_choice("DEFAULT") == MIC_DEFAULT_LABEL
    assert mic_config_to_choice("") == MIC_DEFAULT_LABEL
    # a real device name passes through untouched
    assert mic_choice_to_config("Realtek(R) Audio") == "Realtek(R) Audio"
    assert mic_config_to_choice("Realtek(R) Audio") == "Realtek(R) Audio"


def test_humanize_ts():
    assert humanize_ts("2026-07-03T14:32:10", "2026-07-03") == "Today 14:32"
    assert humanize_ts("2026-07-02T09:05:00", "2026-07-03") == "07-02 09:05"
    assert humanize_ts("", "2026-07-03") == ""
