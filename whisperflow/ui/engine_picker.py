"""Pure view-model helpers for the speech-engine picker — shared by the
Settings "Speech engine" section and the first-run chooser so there's one
place that decides what a provider row says, not two.

No tkinter imports here on purpose: this module is plain data transforms
over whisperflow.stt.providers, unit-testable without a display.
"""

from __future__ import annotations

from whisperflow.stt.providers import Provider, all_providers


def badge_line(provider: Provider) -> str:
    """One-line summary: privacy · cost · quality · speed."""
    privacy = "🔒 Offline" if provider.kind == "local" else "☁ Cloud"
    cost_icon = "💚" if provider.cost_tier == "free" else "💛"
    quality = provider.quality_tier.capitalize()
    return f"{privacy} · {cost_icon} {provider.cost_note} · {quality} · {provider.speed_note}"


def build_rows(recommended_id: str | None = None) -> list[dict]:
    """One row per registered provider, in registry order."""
    return [
        {
            "id": p.id,
            "display_name": p.display_name,
            "badge": badge_line(p),
            "is_recommended": p.id == recommended_id,
        }
        for p in all_providers()
    ]
