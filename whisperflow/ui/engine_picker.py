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


LOCAL_UNAVAILABLE_NOTE = (
    "Not included in this install — get the Full installer for offline on-device mode"
)


def build_rows(recommended_id: str | None = None, local_available: bool = True) -> list[dict]:
    """One row per registered provider, in registry order.

    `local_available=False` (a cloud-only build) marks the Local row
    unavailable with a note, so the UI can show it honestly ('needs the Full
    installer') instead of letting the user pick a dead end. Every cloud
    provider is always available.
    """
    rows = []
    for p in all_providers():
        available = local_available if p.kind == "local" else True
        rows.append(
            {
                "id": p.id,
                "display_name": p.display_name,
                "badge": badge_line(p),
                "is_recommended": p.id == recommended_id,
                "available": available,
                "unavailable_note": "" if available else LOCAL_UNAVAILABLE_NOTE,
            }
        )
    return rows
