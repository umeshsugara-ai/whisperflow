"""Generate assets/app.ico (multi-res) from the tray's idle mic icon, for
use as the icon on the WhisperFlow launcher shortcut."""

from __future__ import annotations

from pathlib import Path

from whisperflow.ui.icons import state_icon

OUT = Path(__file__).resolve().parent.parent / "assets" / "app.ico"


def main() -> None:
    OUT.parent.mkdir(exist_ok=True)
    img = state_icon("idle")
    sizes = [(16, 16), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)]
    img.save(OUT, format="ICO", sizes=sizes)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
