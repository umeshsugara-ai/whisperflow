"""Pillow-drawn tray icons — one per controller state, no asset files."""

from __future__ import annotations

from PIL import Image, ImageDraw

SIZE = 64

# state -> (fill, ring)
PALETTE = {
    "idle": ("#8a8a8a", "#5a5a5a"),
    "recording": ("#e5484d", "#8c2f1f"),
    "processing": ("#f5a623", "#8a6314"),
    "error": ("#4a4a4a", "#e5484d"),
}


def _mic_icon(fill: str, ring: str) -> Image.Image:
    img = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    # outer ring
    d.ellipse((2, 2, SIZE - 2, SIZE - 2), outline=ring, width=4)
    # mic capsule
    d.rounded_rectangle((24, 12, 40, 36), radius=8, fill=fill)
    # mic cradle
    d.arc((18, 22, 46, 44), start=0, end=180, fill=fill, width=4)
    # stem + base
    d.line((32, 44, 32, 50), fill=fill, width=4)
    d.line((24, 52, 40, 52), fill=fill, width=4)
    return img


def state_icon(state: str) -> Image.Image:
    fill, ring = PALETTE.get(state, PALETTE["idle"])
    return _mic_icon(fill, ring)


def all_state_icons() -> dict[str, Image.Image]:
    return {name: state_icon(name) for name in PALETTE}
