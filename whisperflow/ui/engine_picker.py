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


# cost_tier -> (chip text, chip bg, chip fg) — dark-theme colors matching the
# app palette. A native Tk combobox can't color individual dropdown items on
# Windows, so the tier is shown as a bold colored chip in the detail panel
# next to the selection instead.
CHIP_STYLES: dict[str, tuple[str, str, str]] = {
    "free": ("FREE", "#1f4d2e", "#7ee2a0"),
    "freemium": ("FREE TO START", "#4d3a1f", "#f5c778"),
    "paid": ("PAID", "#4d1f24", "#f08a8f"),
}


def cost_chip(provider: Provider) -> dict:
    """Colored cost-tier chip for the picker UIs: {text, bg, fg}."""
    text, bg, fg = CHIP_STYLES[provider.cost_tier]
    return {"text": text, "bg": bg, "fg": fg}


LOCAL_UNAVAILABLE_NOTE = "Not available in this install — pick a free cloud engine instead"


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
