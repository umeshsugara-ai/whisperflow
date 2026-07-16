# -*- coding: utf-8 -*-
"""Pure view-model helpers for the provider picker (Settings + first-run chooser)."""

from whisperflow.stt import providers
from whisperflow.ui.engine_picker import badge_line, build_rows, cost_chip


def test_badge_line_for_cloud_provider_mentions_cost_and_quality():
    groq = providers.get("groq")
    line = badge_line(groq)
    assert "☁ Cloud" in line
    assert groq.cost_note in line
    assert "Better" in line  # groq.quality_tier capitalized
    assert groq.speed_note in line


def test_badge_line_for_local_provider_says_offline():
    local = providers.get("local")
    line = badge_line(local)
    assert "🔒 Offline" in line
    assert local.cost_note in line


def test_build_rows_covers_every_registered_provider_in_order():
    rows = build_rows()
    ids = [r["id"] for r in rows]
    assert ids == [p.id for p in providers.all_providers()]
    for row in rows:
        assert set(row.keys()) == {
            "id", "display_name", "badge", "is_recommended", "available", "unavailable_note",
        }
        assert row["is_recommended"] is False


def test_build_rows_marks_the_recommended_provider():
    rows = build_rows(recommended_id="groq")
    flagged = [r for r in rows if r["is_recommended"]]
    assert len(flagged) == 1
    assert flagged[0]["id"] == "groq"


def test_build_rows_no_recommendation_flags_nothing():
    rows = build_rows(recommended_id=None)
    assert all(not r["is_recommended"] for r in rows)


def test_build_rows_all_available_by_default():
    rows = build_rows()
    assert all(r["available"] for r in rows)
    assert all(r["unavailable_note"] == "" for r in rows)


def test_cost_chip_free_is_green():
    chip = cost_chip(providers.get("groq"))
    assert chip["text"] == "FREE"
    assert chip["bg"] == "#1f4d2e"


def test_cost_chip_freemium_says_free_to_start():
    # NVIDIA/Deepgram give signup credits that run out — not the same as
    # Groq/Gemini's standing free tiers, and the chip must not blur that
    for pid in ("nvidia", "deepgram"):
        chip = cost_chip(providers.get(pid))
        assert chip["text"] == "FREE TO START"
        assert chip["bg"] == "#4d3a1f"


def test_cost_chip_paid_is_highlighted_distinctly():
    chip = cost_chip(providers.get("openai"))
    assert chip["text"] == "PAID"
    # paid must be visually distinct from both free tiers
    free_bg = cost_chip(providers.get("groq"))["bg"]
    freemium_bg = cost_chip(providers.get("nvidia"))["bg"]
    assert chip["bg"] not in (free_bg, freemium_bg)


def test_cost_chip_every_provider_has_a_valid_tier():
    # a new provider with a typo'd cost_tier must fail loudly here, not
    # crash the picker UI at runtime
    for p in providers.all_providers():
        chip = cost_chip(p)
        assert chip["text"] and chip["bg"] and chip["fg"]


def test_build_rows_local_unavailable_marks_only_local():
    rows = build_rows(local_available=False)
    by_id = {r["id"]: r for r in rows}
    assert by_id["local"]["available"] is False
    assert by_id["local"]["unavailable_note"]  # non-empty guidance
    assert by_id["local"]["unavailable_note"] == "Not available in this install — pick a free cloud engine instead"
    # every cloud provider stays available
    for cid in ("groq", "gemini", "openai", "deepgram", "nvidia"):
        assert by_id[cid]["available"] is True
        assert by_id[cid]["unavailable_note"] == ""
