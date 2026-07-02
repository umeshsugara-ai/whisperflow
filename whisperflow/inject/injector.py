"""Tiered text injector — decides how dictated text reaches the focused app.

Policy (config [inject]):
- method = "type":  always SendInput unicode typing (tier 1)
- method = "paste": always clipboard paste with snapshot/restore (tier 2)
- method = "auto":  type for text <= paste_threshold_chars, paste for longer;
                    if typing raises, fall back to paste once.

Wispr Flow ships paste-first and its Windows record shows why that's fragile
(UIPI, WSL, terminal regressions). We invert the order: real unicode key
events first, clipboard only when length makes typing slow.
"""

from __future__ import annotations

import logging

from whisperflow.config import InjectConfig

from . import clipboard, sendinput

log = logging.getLogger(__name__)


class InjectionError(RuntimeError):
    """All applicable injection tiers failed."""


def inject(text: str, cfg: InjectConfig) -> str:
    """Inject `text` into the focused window. Returns the method used."""
    if not text:
        return "none"

    if cfg.method == "type":
        sendinput.type_text(text, interval_ms=cfg.type_interval_ms)
        return "type"

    if cfg.method == "paste":
        clipboard.paste_text(text)
        return "paste"

    # auto
    if len(text) <= cfg.paste_threshold_chars:
        try:
            sendinput.type_text(text, interval_ms=cfg.type_interval_ms)
            return "type"
        except OSError as exc:
            log.warning("type-injection failed (%s); falling back to paste", exc)
            clipboard.paste_text(text)
            return "paste-fallback"
    else:
        clipboard.paste_text(text)
        return "paste"
